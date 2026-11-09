// VMM stress test of the real implementation: the arena and the growable buffers of ggml-cuda/HIP, through the same backend
// registry functions the llama.cpp KV cache uses. A GPU page fault kills the process, so every run is a separate process (vmm-stress.sh).
//   vmm-stress-impl <a|b|c|d> [--chunk-mib N] [--target-gib N] [--sections N]
// The mode is the one of the implementation: GGML_CUDA_VMM_SYNC, GGML_CUDA_VMM_WAIT_MS, GGML_CUDA_VMM_DEBUG.
// Cases (after every growth every buffer is completely written, read and written by a GPU kernel (ADD1), and read back):
//   a  one 9 GiB arena, sections growing in place up to --target-gib in total
//   b  growth; shrink the way the KV does it (new arena, contents copied, old arena freed); growth; then the sections freed and
//      mapped and grown again on the same addresses of the same arena
//   c  the VRAM almost filled with ordinary buffers first, then growth
//   d  several arenas one after the other
// Env: VMM_BACKEND (path of the backend library, default /usr/lib/ollama/rocm_v7_2/libggml-hip.so if it exists),
//      VMM_IMPL_SIM=1 (simulated growable buffers, to test this program on a backend without VMM)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static const size_t MiB = 1ull << 20;

typedef void *                (*arena_new_t)(ggml_backend_buffer_type_t buft, size_t size);
typedef size_t                (*arena_alignment_t)(void * arena);
typedef ggml_backend_buffer_t (*arena_buffer_t)(void * arena, size_t offset, size_t size, size_t max_size);
typedef void                  (*arena_free_t)(void * arena);
typedef bool                  (*arena_commit_t)(void * arena);
typedef bool                  (*buffer_grow_t)(ggml_backend_buffer_t buffer, size_t size);
typedef bool                  (*buffer_grow_commit_t)(ggml_backend_buffer_t buffer);

static arena_new_t          arena_new;
static arena_alignment_t    arena_alignment;
static arena_buffer_t       arena_buffer;
static arena_free_t         arena_free;
static arena_commit_t       arena_commit;
static buffer_grow_t        buffer_grow;
static buffer_grow_commit_t buffer_grow_commit;

// simulated growable buffers: a buffer of the section size
static ggml_backend_buffer_type_t sim_buft;
static void * sim_arena_new(ggml_backend_buffer_type_t buft, size_t) { sim_buft = buft; return (void *) 1; }
static size_t sim_arena_alignment(void *) { return 2*MiB; }
static ggml_backend_buffer_t sim_arena_buffer(void *, size_t, size_t size, size_t max_size) { return ggml_backend_buft_alloc_buffer(sim_buft, std::max(size, max_size)); }
static void sim_arena_free(void *) {}
static bool sim_arena_commit(void *) { return true; }
static bool sim_buffer_grow(ggml_backend_buffer_t b, size_t size) { return size <= ggml_backend_buffer_get_size(b); }
static bool sim_buffer_grow_commit(ggml_backend_buffer_t) { return true; }

[[noreturn]] static void fail(const std::string & msg) {
    printf("FAIL: %s\n", msg.c_str());
    fflush(stdout);
    exit(1);
}

static ggml_backend_dev_t         g_dev;
static ggml_backend_t             g_backend;
static ggml_backend_buffer_type_t g_buft;
static ggml_backend_buffer_t      g_one_buf;
static uint32_t                   g_seed = 1;
static bool                       g_sim  = false;

struct section {
    ggml_backend_buffer_t buf;
    size_t size;
    size_t offset;
    size_t max_size;
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
    printf("  VRAM %-34s used(sysfs) %6zu MiB, device free %6zu MiB\n", what, vram_used()/MiB, dev_free()/MiB);
    fflush(stdout);
}

// write the whole buffer, a GPU kernel reads and writes every element (ADD1 in place), read it back and compare
static void verify(const section & s) {
    const size_t n = s.size/sizeof(float);
    ggml_init_params params = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * t   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    if (ggml_backend_tensor_alloc(s.buf, t, ggml_backend_buffer_get_base(s.buf)) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(g_one_buf, one, ggml_backend_buffer_get_base(g_one_buf)) != GGML_STATUS_SUCCESS) {
        fail("tensor_alloc");
    }
    const float v_one = 1.0f;
    ggml_backend_tensor_set(one, &v_one, 0, sizeof(v_one));

    std::vector<float> data(n);
    const uint32_t seed = g_seed++;
    for (size_t i = 0; i < n; ++i) {
        data[i] = (float) ((i + seed) & 0xffff);
    }
    ggml_backend_tensor_set(t, data.data(), 0, n*sizeof(float));

    ggml_tensor * res = ggml_add1_inplace(ctx, t, one);
    if (ggml_backend_view_init(res) != GGML_STATUS_SUCCESS) {
        fail("view_init");
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res);
    if (ggml_backend_graph_compute(g_backend, gf) != GGML_STATUS_SUCCESS) {
        fail("graph_compute");
    }
    ggml_backend_synchronize(g_backend);

    std::vector<float> back(n);
    ggml_backend_tensor_get(t, back.data(), 0, n*sizeof(float));
    for (size_t i = 0; i < n; ++i) {
        if (back[i] != data[i] + 1.0f) {
            fail("wrong value at element " + std::to_string(i) + " of " + std::to_string(n) + " in the buffer at " +
                 std::to_string((uintptr_t) ggml_backend_buffer_get_base(s.buf)));
        }
    }
    ggml_free(ctx);
}

struct arena {
    void * ptr;
    size_t align;
    size_t next;
};

static arena new_arena(size_t size) {
    void * p = arena_new(g_buft, size);
    if (!p) {
        fail("arena_new returned NULL (no VMM on this device?)");
    }
    return { p, arena_alignment(p), 0 };
}

static section new_section(arena & a, size_t size, size_t max_size) {
    ggml_backend_buffer_t b = arena_buffer(a.ptr, a.next, size, max_size);
    if (!b) {
        fail("arena_buffer returned NULL");
    }
    const size_t offset = a.next;
    a.next += a.align*((max_size + a.align - 1)/a.align);
    return { b, size, offset, max_size };
}

static void commit_new(const arena & a) {
    if (!arena_commit(a.ptr)) {
        fail("arena_commit failed");
    }
}

static void grow_all(std::vector<section> & secs, size_t size) {
    for (auto & s : secs) {
        if (!buffer_grow(s.buf, size)) {
            fail("buffer_grow failed at " + std::to_string(size/MiB) + " MiB");
        }
        s.size = size;
    }
    if (!buffer_grow_commit(secs[0].buf)) {
        fail("buffer_grow_commit failed");
    }
}

// known contents written into from, copied to to by the backend (as the KV shrink does) and checked
static void copy_check(const section & from, const section & to) {
    const size_t n = std::min(from.size, to.size)/sizeof(float);
    ggml_init_params params = { ggml_tensor_overhead()*2, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    if (ggml_backend_tensor_alloc(from.buf, src, ggml_backend_buffer_get_base(from.buf)) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(to.buf, dst, ggml_backend_buffer_get_base(to.buf)) != GGML_STATUS_SUCCESS) {
        fail("tensor_alloc (copy)");
    }
    std::vector<float> data(n), back(n);
    const uint32_t seed = g_seed++;
    for (size_t i = 0; i < n; ++i) {
        data[i] = (float) ((i*3 + seed) & 0xffff);
    }
    ggml_backend_tensor_set(src, data.data(), 0, n*sizeof(float));
    ggml_backend_tensor_copy(src, dst);
    ggml_backend_synchronize(g_backend);
    ggml_backend_tensor_get(dst, back.data(), 0, n*sizeof(float));
    if (memcmp(data.data(), back.data(), n*sizeof(float)) != 0) {
        fail("wrong contents after the copy into the new arena");
    }
    ggml_free(ctx);
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
    commit_new(a);
}

static void verify_all(const std::vector<section> & secs) {
    for (const auto & s : secs) {
        verify(s);
    }
}

static void free_all(std::vector<section> & secs) {
    ggml_backend_synchronize(g_backend);
    for (auto & s : secs) {
        ggml_backend_buffer_free(s.buf);
    }
    secs.clear();
}

// sections growing together from one chunk up to per_section
static void grow_to(std::vector<section> & secs, size_t chunk, size_t per_section) {
    for (size_t size = secs[0].size + chunk; size <= per_section; size += chunk) {
        grow_all(secs, size);
        verify_all(secs);
    }
}

static std::vector<section> start_sections(arena & a, int n, size_t chunk, size_t max_size) {
    std::vector<section> secs;
    for (int i = 0; i < n; ++i) {
        secs.push_back(new_section(a, chunk, max_size));
    }
    commit_new(a);
    verify_all(secs);
    return secs;
}

int main(int argc, char ** argv) {
    if (argc < 2 || strlen(argv[1]) != 1 || argv[1][0] < 'a' || argv[1][0] > 'd') {
        printf("usage: %s <a|b|c|d> [--chunk-mib N] [--target-gib N] [--sections N]\n", argv[0]);
        return 2;
    }
    const char cs = argv[1][0];
    size_t chunk   = 64*MiB;
    size_t target  = 2ull << 30;
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
        arena_commit = sim_arena_commit; buffer_grow = sim_buffer_grow; buffer_grow_commit = sim_buffer_grow_commit;
    } else {
        arena_new          = (arena_new_t)          ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_arena_new");
        arena_alignment    = (arena_alignment_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_arena_alignment");
        arena_buffer       = (arena_buffer_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_arena_buffer");
        arena_free         = (arena_free_t)         ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_arena_free");
        arena_commit       = (arena_commit_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_arena_commit");
        buffer_grow        = (buffer_grow_t)        ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_grow");
        buffer_grow_commit = (buffer_grow_commit_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_grow_commit");
        if (!arena_new || !arena_alignment || !arena_buffer || !arena_free || !arena_commit || !buffer_grow || !buffer_grow_commit) {
            fail("the backend does not provide the growable buffer functions");
        }
    }
    g_one_buf = ggml_backend_buft_alloc_buffer(g_buft, 4096);
    if (!g_one_buf) {
        fail("cannot allocate the small buffer");
    }

    const size_t free0 = dev_free();
    if (free0 > (1ull << 30)) {
        target = std::min(target, free0 - (1ull << 30));
    }
    size_t per_section = target/n_secs;
    per_section -= per_section % chunk;
    if (per_section < chunk) {
        per_section = chunk;
    }
    const char * sync = getenv("GGML_CUDA_VMM_SYNC");
    printf("impl case %c: GGML_CUDA_VMM_SYNC=%s, %s, %d sections up to %zu MiB, chunk %zu MiB\n", cs, sync ? sync : "(auto)",
        ggml_backend_dev_name(g_dev), n_secs, per_section/MiB, chunk/MiB);
    report_vram("at start");
    const size_t vram0 = vram_used();
    const size_t max_size = per_section + chunk;

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
        printf("  first arena: first section at %p\n", ggml_backend_buffer_get_base(secs[0].buf));
        grow_to(secs, chunk, per_section);
        report_vram("grown");
        // shrink the way the KV does it: new sections in a new arena, the contents copied, then the old arena goes
        arena a2 = new_arena(9ull << 30);
        auto secs2 = start_sections(a2, n_secs, chunk, max_size);
        printf("  new arena: first section at %p\n", ggml_backend_buffer_get_base(secs2[0].buf));
        for (int i = 0; i < n_secs; ++i) {
            copy_check(secs[i], secs2[i]);
        }
        free_all(secs);
        arena_free(a.ptr);
        verify_all(secs2);
        report_vram("shrunk (new arena, old one freed)");
        grow_to(secs2, chunk, per_section);
        report_vram("grown again");
        // the same addresses again: unmap, map, grow
        remap_same(a2, secs2, chunk);
        verify_all(secs2);
        report_vram("mapped again on the same addresses");
        grow_to(secs2, chunk, per_section);
        report_vram("grown on the same addresses");
        free_all(secs2);
        arena_free(a2.ptr);
    } else if (cs == 'c') {
        const size_t need = std::min<size_t>(1ull << 30, target);
        size_t per = need/n_secs;
        per -= per % chunk;
        per = std::max(per, chunk);
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
        std::vector<arena> arenas;
        std::vector<std::vector<section>> all;
        // the 8 arenas together stay within the target
        size_t per = std::min<size_t>(target/(8*n_secs), 256*MiB);
        per = std::max(per - per % chunk, chunk);
        for (int i = 0; i < 8; ++i) {
            arenas.push_back(new_arena(2ull << 30));
            all.push_back(start_sections(arenas.back(), n_secs, chunk, per + chunk));
            grow_to(all.back(), chunk, per);
        }
        report_vram("grown");
        for (const auto & secs : all) {
            verify_all(secs);
        }
        for (size_t i = 0; i < all.size(); ++i) {
            free_all(all[i]);
            arena_free(arenas[i].ptr);
        }
    }

    ggml_backend_synchronize(g_backend);
    report_vram("after freeing everything");
    const size_t vram1 = vram_used();
    if (vram0 && vram1 > vram0 + 128*MiB) {
        fail("the VRAM did not come back (" + std::to_string(vram0/MiB) + " -> " + std::to_string(vram1/MiB) + " MiB)");
    }
    printf("impl case %c: OK\n", cs);
    return 0;
}
