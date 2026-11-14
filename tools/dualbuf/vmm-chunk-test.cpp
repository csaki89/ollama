// Build and run on any machine: c++ -std=c++17 -O2 -o vmm-chunk-test tools/dualbuf/vmm-chunk-test.cpp && ./vmm-chunk-test
// CPU check of the VMM chunk bookkeeping (the arithmetic of map_to, ggml_backend_cuda_buffer_shrink and dyn_round, copied),
// with the clr hipMemSetAccess size rule simulated: the size must equal a prefix sum of the arena's mappings in creation order.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <vector>

static const size_t KiB = 1024, MiB = 1024*1024;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

struct mapping { size_t ptr, size; };
struct arena {
    size_t align, chunk;
    std::list<mapping> subs; // the clr parent's sub-buffers, in creation order
};

// clr hipMemSetAccess: accumulate the parent's sub-buffer sizes from the first; > size or never == size -> invalid
static bool clr_set_access_ok(const arena & a, size_t size) {
    size_t acc = 0;
    for (const auto & s : a.subs) {
        acc += s.size;
        if (acc > size) return false;
        if (acc == size) return true;
    }
    return false;
}

struct section {
    arena * a;
    size_t base, size_section, mapped = 0, logical = 0;
    std::vector<mapping> mappings;
    size_t n_map_calls = 0;

    // ggml-cuda map_to
    bool map_to(size_t size) {
        const size_t chunk = a->chunk;
        const size_t need  = chunk * ((size + chunk - 1) / chunk);
        if (need > size_section) return false;
        while (mapped < need) {
            const size_t ptr = base + mapped;
            a->subs.push_back({ ptr, chunk });
            CHECK(clr_set_access_ok(*a, chunk), "set access %zu at section offset %zu", chunk, mapped);
            mappings.push_back({ ptr, chunk });
            mapped += chunk;
            n_map_calls++;
        }
        return true;
    }
    // ggml_backend_cuda_buffer_grow
    bool grow(size_t size) {
        if (size <= logical) return true;
        if (!map_to(size)) return false;
        logical = size;
        return true;
    }
    // ggml_backend_cuda_buffer_shrink
    size_t shrink(size_t size) {
        if (size >= logical) return 0;
        const size_t chunk = a->chunk;
        const size_t keep  = chunk * ((std::max<size_t>(size, 1) + chunk - 1) / chunk);
        size_t freed = 0;
        while (!mappings.empty() && mapped - mappings.back().size >= keep) {
            const auto m = mappings.back();
            a->subs.remove_if([&](const mapping & s) { return s.ptr == m.ptr; });
            mapped -= m.size;
            freed += m.size;
            mappings.pop_back();
        }
        logical = size;
        return freed;
    }
};

// llama_kv_cache::dyn_round
static uint32_t dyn_round(uint32_t n, uint32_t base, uint32_t chunk, uint32_t max) {
    if (n <= base) return base;
    return std::min<uint32_t>(max, base + ((n - base + chunk - 1)/chunk)*chunk);
}

// the VMM step of llama_kv_cache::set_dynamic: the cells of one chunk of the largest section row, a multiple of 256
static uint32_t vmm_step(size_t chunk, size_t row, uint32_t max) {
    const uint32_t cells = (uint32_t) std::min<size_t>(chunk / row, max);
    return std::max<uint32_t>(256, cells - cells % 256);
}

// the chunk choice of ggml_backend_cuda_vmm_arena_new with GGML_CUDA_VMM_CHUNK_KIB (kib), for an alignment that is not raised
static size_t pick_chunk(size_t kib, size_t gran_min, size_t align) {
    size_t chunk = kib << 10;
    if (gran_min > 0) chunk = gran_min * std::max<size_t>(1, (chunk + gran_min - 1) / gran_min);
    if (chunk == 0 || (align % chunk != 0 && chunk % align != 0)) chunk = align;
    return chunk;
}

// a driver that books every allocation rounded up to unit in its free memory (amdgpu KFD: 2 MiB); 0: books the size
static size_t booked(size_t size, size_t unit) {
    return unit ? unit * ((size + unit - 1) / unit) : size;
}

// ggml_cuda_vmm_booking_unit (two probes of a minimal handle) and the chunk without GGML_CUDA_VMM_CHUNK_KIB; align is raised to it
static size_t auto_chunk(size_t gran_min, size_t gran_rec, size_t min_align, size_t driver_unit, size_t & align, size_t & unit) {
    align = std::max(gran_rec, min_align);
    if (gran_min > 0) align = gran_min * ((align + gran_min - 1) / gran_min);
    const size_t d0 = booked(gran_min, driver_unit), d1 = booked(gran_min, driver_unit);
    unit = d0 > 0 && d0 == d1 && d0 % gran_min == 0 ? d0 : 0;
    size_t chunk = align;
    if (unit > 0 && std::lcm(align, unit) <= 32*align) chunk = std::lcm(align, unit);
    align = std::max(align, chunk);
    return chunk;
}

static void run(size_t chunk_kib) {
    const size_t per_cell = 1*KiB;          // qwen3-coder:30b: 4 KV heads x 128 x f16
    const size_t align = 2*MiB;
    arena a { align, pick_chunk(chunk_kib, 4*KiB, align), {} };
    const uint32_t base = 4096, max = 32768, step = vmm_step(a.chunk, per_cell, max);
    const size_t sec_size = align * ((max*per_cell + align - 1) / align);
    std::vector<section> secs;
    for (int i = 0; i < 96; ++i) {
        secs.push_back({ &a, i*sec_size, sec_size });
    }
    // the initial mapping (arena_buffer: map_to(size))
    for (auto & s : secs) {
        CHECK(s.map_to(base*per_cell), "initial map");
        s.logical = base*per_cell;
    }
    auto check_all = [&](const char * what) {
        for (auto & s : secs) {
            const size_t want = a.chunk * ((s.logical + a.chunk - 1) / a.chunk);
            CHECK(s.mapped == want, "%s: mapped %zu, want %zu (logical %zu)", what, s.mapped, want, s.logical);
            CHECK(s.mapped <= s.size_section, "%s: over the section", what);
            for (const auto & m : s.mappings) CHECK(m.size == a.chunk, "%s: mapping of %zu", what, m.size);
        }
    };
    check_all("initial");
    printf("chunk %zu KiB: initial %zu MiB mapped per section\n", a.chunk/KiB, secs[0].mapped/MiB);

    // growth as the KV does it: the steps from the starting size, every layer K and V by the same cells
    uint32_t n = base;
    std::vector<uint32_t> sizes;
    while (n < 13000) {
        const uint32_t n_new = dyn_round(n + 1, base, step, max);
        CHECK(n_new - n == step, "step %u -> %u is not %u", n, n_new, step);
        size_t calls0 = 0;
        for (auto & s : secs) calls0 += s.n_map_calls;
        for (auto & s : secs) CHECK(s.grow((size_t) n_new*per_cell), "grow %u", n_new);
        size_t calls = 0;
        for (auto & s : secs) calls += s.n_map_calls;
        const size_t want_new = (a.chunk * (((size_t) n_new*per_cell + a.chunk - 1)/a.chunk) - a.chunk * (((size_t) n*per_cell + a.chunk - 1)/a.chunk)) / a.chunk;
        CHECK(want_new <= 1 || a.chunk/per_cell < 256, "more than one chunk per section in a step");
        CHECK(calls - calls0 == want_new*secs.size(), "grow %u -> %u: %zu map calls, want %zu", n, n_new, calls - calls0, want_new*secs.size());
        sizes.push_back(n_new);
        n = n_new;
    }
    check_all("grown");
    printf("  grown %u -> %u cells in %zu steps of %u; per section logical %zu KiB, mapped %zu KiB\n", base, n, sizes.size(), step,
        secs[0].logical/KiB, secs[0].mapped/KiB);

    // shrink in place to the starting size: only the whole chunks above it go
    size_t freed = 0, mapped_before = 0;
    for (auto & s : secs) mapped_before += s.mapped;
    for (auto & s : secs) freed += s.shrink((size_t) base*per_cell);
    check_all("shrunk");
    size_t mapped_after = 0;
    for (auto & s : secs) mapped_after += s.mapped;
    CHECK(mapped_before - mapped_after == freed, "freed bookkeeping");
    printf("  shrunk to %u cells: %zu MiB unmapped, %zu MiB left mapped\n", base, freed/MiB, mapped_after/MiB);

    // a shrink to a size inside a chunk keeps that chunk
    for (auto & s : secs) s.grow((size_t) (base + step)*per_cell);
    for (auto & s : secs) s.shrink((size_t) (base + 100)*per_cell);
    check_all("partial shrink");

    // grow again (the clr rule is checked on every mapping)
    n = base + 100;
    for (int i = 0; i < 6; ++i) {
        const uint32_t n_new = dyn_round(n + 1, base, step, max);
        for (auto & s : secs) CHECK(s.grow((size_t) n_new*per_cell), "regrow %u", n_new);
        n = n_new;
    }
    check_all("grown again");
    // to the max and over it
    for (auto & s : secs) CHECK(s.grow((size_t) max*per_cell), "grow to max");
    CHECK(dyn_round(max + 5, base, step, max) == max, "dyn_round over the max");
    CHECK(!secs[0].grow((size_t) max*per_cell + a.chunk), "growing over the section must fail");
    check_all("max");
}

int main() {
    // the old scheme for comparison: a 4 MiB first mapping, then a 2 MiB one -> the clr rule fails
    {
        arena a { 2*MiB, 0, {} };
        a.subs.push_back({ 0, 4*MiB });
        a.subs.push_back({ 4*MiB, 2*MiB });
        printf("old scheme (4 MiB first, then 2 MiB): set access %s\n", clr_set_access_ok(a, 2*MiB) ? "ok" : "hipErrorInvalidValue");
    }
    CHECK(pick_chunk(2048, 4*KiB, 2*MiB) == 2*MiB, "default chunk (the alignment)");
    CHECK(pick_chunk(1024, 4*KiB, 2*MiB) == 1*MiB, "1 MiB chunk");
    CHECK(pick_chunk(512, 4*KiB, 2*MiB) == 512*KiB, "512 KiB chunk");
    CHECK(pick_chunk(3000, 4*KiB, 2*MiB) == 2*MiB, "a chunk that does not divide the alignment");
    CHECK(pick_chunk(1024, 2*MiB, 2*MiB) == 2*MiB, "minimum granularity 2 MiB");
    // the automatic chunk: the measured booking unit decides, the driver's booking then equals the mapped size
    struct { size_t gmin, grec, drv, want_chunk, want_unit; const char * what; } cases[] = {
        { 4*KiB, 4*KiB,  2*MiB, 2*MiB, 2*MiB, "RX 6700 / amdgpu KFD: granularity 4 KiB, booking 2 MiB" },
        { 4*KiB, 4*KiB,  0,     2*MiB, 4*KiB, "a driver that books the size: the alignment" },
        { 4*KiB, 4*KiB,  8*MiB, 8*MiB, 8*MiB, "a larger booking unit raises the chunk and the alignment" },
        { 2*MiB, 2*MiB,  2*MiB, 2*MiB, 2*MiB, "CUDA-like: granularity 2 MiB" },
        { 4*KiB, 4*KiB,  3*MiB, 6*MiB, 3*MiB, "a booking unit that is not a power of two: the common multiple" },
    };
    for (const auto & c : cases) {
        size_t align = 0, unit = 0;
        const size_t chunk = auto_chunk(c.gmin, c.grec, 2*MiB, c.drv, align, unit);
        CHECK(chunk == c.want_chunk && unit == c.want_unit, "%s: chunk %zu, unit %zu", c.what, chunk, unit);
        CHECK(align % chunk == 0 && chunk % c.gmin == 0, "%s: chunk %zu, align %zu", c.what, chunk, align);
        // 96 sections grown by 5 steps of one chunk: the driver books exactly what is mapped
        const size_t mapped = 96*5*chunk, drv = 96*5*booked(chunk, c.drv);
        CHECK(mapped == drv, "%s: mapped %zu MiB, booked by the driver %zu MiB", c.what, mapped/MiB, drv/MiB);
        printf("auto chunk: %s -> unit %zu KiB, chunk %zu KiB, alignment %zu KiB\n", c.what, unit/KiB, chunk/KiB, align/KiB);
    }
    // the old 1 MiB chunk on the KFD: the driver books double (the 1918 -> 190 MiB on the NAS for 864 MiB)
    CHECK(96*9*booked(1*MiB, 2*MiB) == 2*96*9*1*MiB, "1 MiB chunks are booked double");
    CHECK(vmm_step(2*MiB, 1*KiB, 32768) == 2048, "qwen3-coder:30b step with the default chunk (the 2 MiB alignment)");
    CHECK(vmm_step(1*MiB, 1*KiB, 32768) == 1024, "qwen3-coder:30b step with GGML_CUDA_VMM_CHUNK_KIB=1024");
    CHECK(vmm_step(1*MiB, 3*KiB, 32768) == 256, "3 KiB rows: 341 cells -> 256");
    CHECK(vmm_step(1*MiB, 8*KiB, 32768) == 256, "8 KiB rows: 128 cells -> at least 256 (two chunks a step)");
    CHECK(vmm_step(2*MiB, 512, 32768) == 4096, "512 B rows, 2 MiB chunk");
    run(1024);
    run(512);
    run(2048);
    printf(fails ? "FAILED: %d\n" : "all OK\n", fails);
    return fails ? 1 : 0;
}
