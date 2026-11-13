// VMM stress test of the real implementation: the arena and the growable buffers of ggml-cuda/HIP, through the same backend
// registry functions the llama.cpp KV cache uses. A GPU page fault kills the process, so every run is a separate process (vmm-stress.sh).
//   vmm-stress-impl <a|b|c|d> [--chunk-mib N] [--target-gib N] [--sections N]
// The mode is the one of the implementation: GGML_CUDA_VMM_SYNC, GGML_CUDA_VMM_WAIT_MS (GGML_CUDA_VMM_DEBUG must be off: logging delays).
// Every growth step: all sections grow by one chunk (mapping only), one commit (sync, then its zeroing is the first access, only of
// the new chunks), then one word per 4 KiB page of the new chunks is written, and one word per page of the whole sections (old
// and new) is gathered on the GPU and read back (4 bytes per page). The mapping end -> first access time is measured by the backend.
// Cases:
//   a  sections in one 9 GiB arena growing up to --target-gib in total
//   b  growth; shrink the way the KV does it (new arena, contents copied, old arena freed); growth; then the sections freed,
//      mapped again on the same addresses of the same arena, and grown again
//   c  the VRAM almost filled with ordinary buffers first, then growth
//   d  8 arenas one after the other
// At the end: no arena and no mapped byte may be left (backend counters), and the VRAM must come back (sysfs, polled for 3 s).
// Env: VMM_BACKEND (path of the backend library, default /usr/lib/ollama/rocm_v7_2/libggml-hip.so if it exists),
//      VMM_IMPL_SIM=1 (simulated growable buffers, to test this program on a backend without VMM)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

static const size_t MiB  = 1ull << 20;
static const size_t PAGE = 4096;

typedef void *                (*arena_new_t)(ggml_backend_buffer_type_t buft, size_t size);
typedef size_t                (*arena_alignment_t)(void * arena);
typedef ggml_backend_buffer_t (*arena_buffer_t)(void * arena, size_t offset, size_t size, size_t max_size);
typedef void                  (*arena_free_t)(void * arena);
typedef bool                  (*arena_commit_t)(void * arena);
typedef bool                  (*buffer_grow_t)(ggml_backend_buffer_t buffer, size_t size);
typedef bool                  (*buffer_grow_commit_t)(ggml_backend_buffer_t buffer);
typedef void                  (*vmm_stats_t)(int64_t * stats);

static arena_new_t          arena_new;
static arena_alignment_t    arena_alignment;
static arena_buffer_t       arena_buffer;
static arena_free_t         arena_free;
static arena_commit_t       arena_commit;
static buffer_grow_t        buffer_grow;
static buffer_grow_commit_t buffer_grow_commit;
static vmm_stats_t          vmm_stats;

// simulated growable buffers: a buffer of the section size
static ggml_backend_buffer_type_t sim_buft;
static void * sim_arena_new(ggml_backend_buffer_type_t buft, size_t) { sim_buft = buft; return (void *) 1; }
static size_t sim_arena_alignment(void *) { return 2*MiB; }
static ggml_backend_buffer_t sim_arena_buffer(void *, size_t, size_t size, size_t max_size) { return ggml_backend_buft_alloc_buffer(sim_buft, std::max(size, max_size)); }
static void sim_arena_free(void *) {}
static bool sim_arena_commit(void *) { return true; }
static bool sim_buffer_grow(ggml_backend_buffer_t b, size_t size) { return size <= ggml_backend_buffer_get_size(b); }
static bool sim_buffer_grow_commit(ggml_backend_buffer_t) { return true; }
static void sim_vmm_stats(int64_t * stats) { memset(stats, 0, 4*sizeof(int64_t)); }

[[noreturn]] static void fail(const std::string & msg) {
    printf("FAIL: %s\n", msg.c_str());
    fflush(stdout);
    exit(1);
}

static ggml_backend_dev_t         g_dev;
static ggml_backend_t             g_backend;
static ggml_backend_buffer_type_t g_buft;
static ggml_backend_buffer_t      g_scratch;   // ones, then the gathered page words of every section
static size_t                     g_max_pages; // per section
static bool                       g_sim = false;
static int64_t                    g_gap_max   = 0; // mapping end -> first access (us)
static int64_t                    g_sync_max  = 0; // of which the sync
static int64_t                    g_other_max = 0; // the rest (must be ~0)

struct section {
    ggml_backend_buffer_t buf;
    size_t size;
    size_t offset;
    size_t max_size;
};

struct arena {
    void * ptr;
    size_t align;
    size_t next;
};

static size_t vram_used() {
    size_t best_total = 0, used = 0;
    for (int c = 0; c < 16; ++c) {
        const std::string base = "/sys/class/drm/card" + std::to_string(c) + "/device/";
        std::ifstream ft(base + "mem_info_vram_total"), fu(base + "mem_info_vram_used");
        size_t t = 0, u = 0;
        if ((ft >> t) && (fu >> u) && t > best_total) {
            best_total = t;
            used = u;
        }
    }
    return used;
}

static size_t dev_free() {
    size_t free_b = 0, total_b = 0;
    ggml_backend_dev_memory(g_dev, &free_b, &total_b);
    return free_b;
}

static void report_vram(const char * what) {
    printf("  VRAM %-36s used(sysfs) %6zu MiB, device free %6zu MiB\n", what, vram_used()/MiB, dev_free()/MiB);
    fflush(stdout);
}

// poll the sysfs VRAM until it is at most limit (the kernel frees asynchronously); the time it took in ms, -1 if not within 3 s
static int wait_vram_below(size_t limit) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const int ms = (int) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        if (vram_used() <= limit) {
            return ms;
        }
        if (ms > 3000) {
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

static void note_gap() {
    int64_t st[4];
    vmm_stats(st);
    g_gap_max   = std::max(g_gap_max, st[2]);
    g_sync_max  = std::max(g_sync_max, st[3]);
    g_other_max = std::max(g_other_max, st[2] - st[3]);
}

static void init_views(ggml_cgraph * gf) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        for (int j = -1; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * t = j < 0 ? node : node->src[j];
            if (t && !t->buffer && t->view_src && ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                fail("view_init");
            }
        }
    }
}

// one word per page of [from, size) set to 1 in every section, then one word per page of [0, size) gathered and checked (all 1)
static void mark_and_check(const std::vector<section> & secs, const std::vector<size_t> & from) {
    const size_t n = secs.size();
    ggml_init_params params = { ggml_tensor_overhead()*(8*n + 8) + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * ones = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, g_max_pages);
    ggml_tensor * samp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, g_max_pages*n);
    char * sbase = (char *) ggml_backend_buffer_get_base(g_scratch);
    if (ggml_backend_tensor_alloc(g_scratch, ones, sbase) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(g_scratch, samp, sbase + g_max_pages*sizeof(float)) != GGML_STATUS_SUCCESS) {
        fail("tensor_alloc (scratch)");
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> t(n), dst(n);
    for (size_t i = 0; i < n; ++i) {
        if (secs[i].size/PAGE > g_max_pages) {
            fail("scratch too small");
        }
        t[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, secs[i].size/sizeof(float));
        if (ggml_backend_tensor_alloc(secs[i].buf, t[i], ggml_backend_buffer_get_base(secs[i].buf)) != GGML_STATUS_SUCCESS) {
            fail("tensor_alloc");
        }
        if (from[i] < secs[i].size) {
            const size_t n_new = (secs[i].size - from[i])/PAGE;
            ggml_tensor * v_new = ggml_view_2d(ctx, t[i], 1, n_new, PAGE, from[i]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_view_1d(ctx, ones, n_new, 0), v_new));
        }
    }
    for (size_t i = 0; i < n; ++i) {
        const size_t pages = secs[i].size/PAGE;
        ggml_tensor * v_all = ggml_view_2d(ctx, t[i], 1, pages, PAGE, 0);
        dst[i] = ggml_view_1d(ctx, samp, pages, i*g_max_pages*sizeof(float));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, v_all, dst[i]));
    }
    init_views(gf);
    if (ggml_backend_graph_compute(g_backend, gf) != GGML_STATUS_SUCCESS) {
        fail("graph_compute");
    }
    std::vector<float> back(g_max_pages);
    for (size_t i = 0; i < n; ++i) {
        const size_t pages = secs[i].size/PAGE;
        ggml_backend_tensor_get(dst[i], back.data(), 0, pages*sizeof(float));
        for (size_t p = 0; p < pages; ++p) {
            if (back[p] != 1.0f) {
                fail("wrong word in page " + std::to_string(p) + " of " + std::to_string(pages) + " of section " + std::to_string(i) +
                     " (" + std::to_string(back[p]) + ", new pages from " + std::to_string(from[i]/PAGE) + ")");
            }
        }
    }
    ggml_free(ctx);
}

static arena new_arena(size_t size) {
    void * p = arena_new(g_buft, size);
    if (!p) {
        fail("arena_new returned NULL (no VMM on this device?)");
    }
    return { p, arena_alignment(p), 0 };
}

// new sections mapped with chunk bytes, committed (first access), marked and checked (or left zeroed)
static std::vector<section> start_sections(arena & a, int n, size_t chunk, size_t max_size, bool mark = true) {
    std::vector<section> secs;
    for (int i = 0; i < n; ++i) {
        ggml_backend_buffer_t b = arena_buffer(a.ptr, a.next, chunk, max_size);
        if (!b) {
            fail("arena_buffer returned NULL");
        }
        secs.push_back({ b, chunk, a.next, max_size });
        a.next += a.align*((max_size + a.align - 1)/a.align);
    }
    if (!arena_commit(a.ptr)) {
        fail("arena_commit failed");
    }
    note_gap();
    if (mark) {
        mark_and_check(secs, std::vector<size_t>(secs.size(), 0));
    }
    return secs;
}

// all sections grow by one chunk per step (mapping, then one commit) up to per_section
static void grow_to(std::vector<section> & secs, size_t chunk, size_t per_section) {
    std::vector<size_t> from(secs.size());
    for (size_t size = secs[0].size + chunk; size <= per_section; size += chunk) {
        for (size_t i = 0; i < secs.size(); ++i) {
            from[i] = secs[i].size;
            if (!buffer_grow(secs[i].buf, size)) {
                fail("buffer_grow failed at " + std::to_string(size/MiB) + " MiB");
            }
            secs[i].size = size;
        }
        if (!buffer_grow_commit(secs[0].buf)) {
            fail("buffer_grow_commit failed");
        }
        note_gap();
        mark_and_check(secs, from);
    }
}

static void free_all(std::vector<section> & secs) {
    ggml_backend_synchronize(g_backend);
    for (auto & s : secs) {
        ggml_backend_buffer_free(s.buf);
    }
    secs.clear();
}

// unmap the sections and map them again at the same addresses of the same arena
static void remap_same(arena & a, std::vector<section> & secs, size_t size) {
    ggml_backend_synchronize(g_backend);
    for (auto & s : secs) {
        void * base = ggml_backend_buffer_get_base(s.buf);
        ggml_backend_buffer_free(s.buf);
        s.buf = arena_buffer(a.ptr, s.offset, size, s.max_size);
        if (!s.buf) {
            fail("arena_buffer returned NULL when mapping again");
        }
        if (!g_sim && ggml_backend_buffer_get_base(s.buf) != base) {
            fail("the section came back on a different address");
        }
        s.size = size;
    }
    if (!arena_commit(a.ptr)) {
        fail("arena_commit failed");
    }
    note_gap();
    mark_and_check(secs, std::vector<size_t>(secs.size(), 0));
}

int main(int argc, char ** argv) {
    if (argc < 2 || strlen(argv[1]) != 1 || argv[1][0] < 'a' || argv[1][0] > 'd') {
        printf("usage: %s <a|b|c|d> [--chunk-mib N] [--target-gib N] [--sections N]\n", argv[0]);
        return 2;
    }
    const char cs = argv[1][0];
    size_t chunk   = 64*MiB;
    size_t target  = 6ull << 30;
    int    n_secs  = 4;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--chunk-mib") { chunk = (size_t) atoi(argv[i + 1])*MiB; }
        else if (a == "--target-gib") { target = (size_t) atoi(argv[i + 1]) << 30; }
        else if (a == "--sections") { n_secs = atoi(argv[i + 1]); }
        else { printf("unknown option %s\n", a.c_str()); return 2; }
    }

    std::string path = getenv("VMM_BACKEND") ? getenv("VMM_BACKEND") : "";
    if (path.empty() && std::ifstream("/usr/lib/ollama/rocm_v7_2/libggml-hip.so").good()) {
        path = "/usr/lib/ollama/rocm_v7_2/libggml-hip.so";
    }
    ggml_backend_reg_t reg = nullptr;
    if (!path.empty()) {
        reg = ggml_backend_load(path.c_str());
        if (!reg) {
            fail("cannot load the backend " + path);
        }
        if (ggml_backend_reg_dev_count(reg) == 0) {
            fail("no device in " + path);
        }
        g_dev = ggml_backend_reg_dev_get(reg, 0);
    } else {
        g_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!g_dev) {
            g_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        }
        if (!g_dev) {
            fail("no GPU device");
        }
        reg = ggml_backend_dev_backend_reg(g_dev);
    }
    g_backend = ggml_backend_dev_init(g_dev, nullptr);
    g_buft    = ggml_backend_dev_buffer_type(g_dev);
    if (!g_backend || !g_buft) {
        fail("cannot initialize the device");
    }

    g_sim = getenv("VMM_IMPL_SIM") && atoi(getenv("VMM_IMPL_SIM")) != 0;
    if (g_sim) {
        arena_new = sim_arena_new; arena_alignment = sim_arena_alignment; arena_buffer = sim_arena_buffer; arena_free = sim_arena_free;
        arena_commit = sim_arena_commit; buffer_grow = sim_buffer_grow; buffer_grow_commit = sim_buffer_grow_commit; vmm_stats = sim_vmm_stats;
    } else {
        arena_new          = (arena_new_t)          ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_arena_new");
        arena_alignment    = (arena_alignment_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_arena_alignment");
        arena_buffer       = (arena_buffer_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_arena_buffer");
        arena_free         = (arena_free_t)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_arena_free");
        arena_commit       = (arena_commit_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_arena_commit");
        buffer_grow        = (buffer_grow_t)        ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_grow");
        buffer_grow_commit = (buffer_grow_commit_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_grow_commit");
        vmm_stats          = (vmm_stats_t)          ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_stats");
        if (!arena_new || !arena_alignment || !arena_buffer || !arena_free || !arena_commit || !buffer_grow || !buffer_grow_commit || !vmm_stats) {
            fail("the backend does not provide the growable buffer functions");
        }
    }

    const size_t free0 = dev_free();
    if (free0 > (1ull << 30)) {
        target = std::min<size_t>(target, free0 - (1ull << 30));
    }
    // one cpy of a section stays below 2 GiB
    size_t per_section = std::min<size_t>(target/n_secs, 1536*MiB);
    per_section = std::max(per_section - per_section % chunk, chunk);
    const size_t max_size = per_section + chunk;
    g_max_pages = max_size/PAGE;
    g_scratch = ggml_backend_buft_alloc_buffer(g_buft, (1 + n_secs)*g_max_pages*sizeof(float) + 4096);
    if (!g_scratch) {
        fail("cannot allocate the scratch buffer");
    }
    {
        std::vector<float> ones(g_max_pages, 1.0f);
        ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, g_max_pages);
        if (ggml_backend_tensor_alloc(g_scratch, t, ggml_backend_buffer_get_base(g_scratch)) != GGML_STATUS_SUCCESS) {
            fail("tensor_alloc (ones)");
        }
        ggml_backend_tensor_set(t, ones.data(), 0, g_max_pages*sizeof(float));
        ggml_free(ctx);
    }
    ggml_backend_synchronize(g_backend);

    const char * sync = getenv("GGML_CUDA_VMM_SYNC");
    printf("impl case %c: GGML_CUDA_VMM_SYNC=%s, %s, %d sections up to %zu MiB, chunk %zu MiB\n", cs, sync ? sync : "(auto)",
        ggml_backend_dev_name(g_dev), n_secs, per_section/MiB, chunk/MiB);
    report_vram("at start");
    const size_t vram0 = vram_used();
    const auto t_start = std::chrono::steady_clock::now();

    if (cs == 'a') {
        arena a = new_arena(9ull << 30);
        auto secs = start_sections(a, n_secs, chunk, max_size);
        grow_to(secs, chunk, per_section);
        report_vram("grown");
        free_all(secs);
        arena_free(a.ptr);
    } else if (cs == 'b') {
        arena a = new_arena(9ull << 30);
        auto secs = start_sections(a, n_secs, chunk, max_size);
        grow_to(secs, chunk, per_section);
        report_vram("grown");
        const size_t vram_grown = vram_used();
        // shrink the way the KV does it: new sections in a new arena, the first chunk copied, then the old arena goes
        arena a2 = new_arena(9ull << 30);
        auto secs2 = start_sections(a2, n_secs, chunk, max_size, false);
        {
            ggml_init_params params = { ggml_tensor_overhead()*2*n_secs, nullptr, true };
            ggml_context * ctx = ggml_init(params);
            for (int i = 0; i < n_secs; ++i) {
                ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, chunk/sizeof(float));
                ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, chunk/sizeof(float));
                if (ggml_backend_tensor_alloc(secs[i].buf, src, ggml_backend_buffer_get_base(secs[i].buf)) != GGML_STATUS_SUCCESS ||
                    ggml_backend_tensor_alloc(secs2[i].buf, dst, ggml_backend_buffer_get_base(secs2[i].buf)) != GGML_STATUS_SUCCESS) {
                    fail("tensor_alloc (copy)");
                }
                ggml_backend_tensor_copy(src, dst);
            }
            ggml_free(ctx);
        }
        ggml_backend_synchronize(g_backend);
        free_all(secs);
        arena_free(a.ptr);
        mark_and_check(secs2, std::vector<size_t>(secs2.size(), chunk)); // the copied words (zero if not copied)
        const size_t freed = vram_grown - std::min(vram_grown, vram0);
        const int ms = vram0 ? wait_vram_below(vram_grown - freed/2) : 0;
        if (ms < 0) {
            report_vram("shrunk (new arena, old one freed)");
            fail("the VRAM of the freed arena did not come back within 3 s");
        }
        printf("  shrink: the VRAM of the old arena came back in %d ms\n", ms);
        report_vram("shrunk (new arena, old one freed)");
        grow_to(secs2, chunk, per_section);
        report_vram("grown again");
        remap_same(a2, secs2, chunk);
        grow_to(secs2, chunk, per_section);
        report_vram("grown on the same addresses");
        free_all(secs2);
        arena_free(a2.ptr);
    } else if (cs == 'c') {
        const size_t need = std::min<size_t>(1ull << 30, target);
        size_t per = need/n_secs;
        per = std::max(per - per % chunk, chunk);
        std::vector<ggml_backend_buffer_t> filler;
        size_t filled = 0;
        while (dev_free() > need + 768*MiB) {
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(g_buft, 256*MiB);
            if (!b) {
                break;
            }
            ggml_backend_buffer_clear(b, 1);
            filler.push_back(b);
            filled += 256*MiB;
        }
        ggml_backend_synchronize(g_backend);
        printf("  filler: %zu MiB\n", filled/MiB);
        report_vram("after the filler");
        arena a = new_arena(9ull << 30);
        auto secs = start_sections(a, n_secs, chunk, per + chunk);
        grow_to(secs, chunk, per);
        report_vram("grown");
        free_all(secs);
        arena_free(a.ptr);
        for (auto * b : filler) {
            ggml_backend_buffer_free(b);
        }
    } else {
        // the 8 arenas together stay within the target
        size_t per = std::min<size_t>(target/(8*n_secs), 256*MiB);
        per = std::max(per - per % chunk, chunk);
        std::vector<arena> arenas;
        std::vector<std::vector<section>> all;
        for (int i = 0; i < 8; ++i) {
            arenas.push_back(new_arena(2ull << 30));
            all.push_back(start_sections(arenas.back(), n_secs, chunk, per + chunk));
            grow_to(all.back(), chunk, per);
        }
        report_vram("grown");
        for (const auto & secs : all) {
            mark_and_check(secs, std::vector<size_t>(secs.size(), secs[0].size));
        }
        for (size_t i = 0; i < all.size(); ++i) {
            free_all(all[i]);
            arena_free(arenas[i].ptr);
        }
    }

    const double run_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("  mapping end -> first access: max %.2f ms, of which the sync %.2f ms, anything else at most %.3f ms\n",
        g_gap_max/1000.0, g_sync_max/1000.0, g_other_max/1000.0);
    ggml_backend_synchronize(g_backend);
    int64_t st[4];
    vmm_stats(st);
    if (st[0] != 0 || st[1] != 0) {
        fail("left over in the backend: " + std::to_string(st[0]) + " arenas, " + std::to_string(st[1]) + " mapped bytes");
    }
    const int ms = vram0 ? wait_vram_below(vram0 + 128*MiB) : 0;
    report_vram("after freeing everything");
    if (ms < 0) {
        fail("the VRAM did not come back within 3 s (" + std::to_string(vram0/MiB) + " -> " + std::to_string(vram_used()/MiB) + " MiB)");
    }
    printf("  freed: no arena and no mapping left, the VRAM came back in %d ms; run time %.1f s\n", ms, run_s);
    printf("impl case %c: OK\n", cs);
    return 0;
}
