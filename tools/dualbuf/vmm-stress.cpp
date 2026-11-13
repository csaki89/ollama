// VMM stress test for ROCm (HIP, one source file), the bare pattern without our code. A GPU page fault kills the process,
// so every run is a separate process (vmm-stress.sh).
//   vmm-stress <case> [--sync none|wait|barrier|both] [--wait-ms N] [--chunk-mib N] [--batch N] [--target-gib N] [--barrier-mib N]
// Every growth step maps --batch chunks (like vmm-stress-impl: one chunk per section, 4 sections), syncs once, and the first GPU
// access right after it is one word per 4 KiB page of the new chunks only (everything prepared before the mapping). Then one word
// per page of every chunk (old and new) is checked on the GPU. The mapping end -> first access time is measured.
// Cases (as in vmm-stress-impl):
//   a  one 9 GiB address range, growth up to --target-gib
//   b  growth, shrink (unmap the upper half), growth again on the same addresses; then the range freed and a new one grown
//   c  the VRAM almost filled with hipMalloc first, then growth
//   d  96 separate address ranges of 32 MiB, one chunk each
// Sync after mapping and before the first access:
//   none     nothing
//   wait     sleep --wait-ms
//   barrier  map a fresh allocation of --barrier-mib through KFD (hipMalloc + hipFree), which waits for the page table updates queued before it
//   both     barrier, then wait
// Build: hipcc -O2 --offload-arch=gfx1030 -o vmm-stress vmm-stress.cpp

#include <hip/hip_runtime.h>

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

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("FAIL %s: %s (%d) at line %d\n", #x, hipGetErrorString(e_), (int) e_, __LINE__); fflush(stdout); exit(1); } } while (0)

static const size_t MiB  = 1ull << 20;
static const size_t PAGE = 4096;

typedef std::chrono::steady_clock clk;

// one word per page: word = page index ^ seed
__global__ void k_mark(char * base, size_t n_pages, uint32_t seed) {
    for (size_t i = blockIdx.x*(size_t) blockDim.x + threadIdx.x; i < n_pages; i += (size_t) gridDim.x*blockDim.x) {
        *(uint32_t *) (base + i*PAGE) = (uint32_t) i ^ seed;
    }
}

__global__ void k_check(const char * base, size_t n_pages, uint32_t seed, unsigned long long * bad) {
    for (size_t i = blockIdx.x*(size_t) blockDim.x + threadIdx.x; i < n_pages; i += (size_t) gridDim.x*blockDim.x) {
        if (*(const uint32_t *) (base + i*PAGE) != ((uint32_t) i ^ seed)) {
            atomicAdd(bad, 1ull);
        }
    }
}

struct chunk {
    char * ptr;
    size_t size;
    hipMemGenericAllocationHandle_t handle;
    uint32_t seed;
};

static std::string sync_mode = "both";
static int    wait_ms      = 5;
static size_t barrier_size = 4*MiB;
static size_t align_       = 2*MiB;
static int    dev          = 0;
static unsigned long long * bad_counter = nullptr;
static double gap_max   = 0; // mapping end -> first access (ms)
static double sync_max  = 0; // of which the sync
static double other_max = 0; // the rest (must be ~0)

// VRAM in use, from sysfs (the card with the most VRAM); 0 if not readable
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

static void report_vram(const char * what) {
    size_t free_b = 0, total_b = 0;
    CHECK(hipMemGetInfo(&free_b, &total_b));
    printf("  VRAM %-36s used(sysfs) %6zu MiB, hipMemGetInfo free %6zu of %6zu MiB\n", what, vram_used()/MiB, free_b/MiB, total_b/MiB);
    fflush(stdout);
}

// poll the sysfs VRAM until it is at most limit; the time it took in ms, -1 if not within 3 s
static int wait_vram_below(size_t limit) {
    const auto t0 = clk::now();
    for (;;) {
        const int ms = (int) std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
        if (vram_used() <= limit) {
            return ms;
        }
        if (ms > 3000) {
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

static void wait_mapped() {
    if (sync_mode == "barrier" || sync_mode == "both") {
        void * p = nullptr;
        if (hipMalloc(&p, barrier_size) == hipSuccess) {
            (void) hipFree(p);
        } else {
            (void) hipGetLastError();
        }
    }
    if (sync_mode == "wait" || sync_mode == "both") {
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    }
}

static chunk map_chunk(char * ptr, size_t size, uint32_t seed) {
    hipMemAllocationProp prop = {};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = dev;
    hipMemGenericAllocationHandle_t h;
    CHECK(hipMemCreate(&h, size, &prop, 0));
    CHECK(hipMemMap(ptr, size, 0, h, 0));
    hipMemAccessDesc access = {};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id   = dev;
    access.flags         = hipMemAccessFlagsProtReadWrite;
    CHECK(hipMemSetAccess(ptr, size, &access, 1));
    return { ptr, size, h, seed };
}

static void unmap_chunk(const chunk & c) {
    CHECK(hipMemUnmap(c.ptr, c.size));
    CHECK(hipMemRelease(c.handle));
}

// one word per page of every chunk, on the GPU
static void check_all(const std::vector<chunk> & chunks, const char * what) {
    CHECK(hipMemset(bad_counter, 0, sizeof(*bad_counter)));
    for (const auto & c : chunks) {
        k_check<<<256, 256>>>(c.ptr, c.size/PAGE, c.seed, bad_counter);
    }
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    unsigned long long bad = 0;
    CHECK(hipMemcpy(&bad, bad_counter, sizeof(bad), hipMemcpyDeviceToHost));
    if (bad != 0) {
        printf("FAIL: %llu wrong page words (%s)\n", bad, what);
        fflush(stdout);
        exit(1);
    }
}

// map the chunks at the given addresses, sync once, then immediately mark their pages (the first access), then check everything
static void map_batch(std::vector<chunk> & chunks, const std::vector<char *> & ptrs, size_t chunk_size, uint32_t seed) {
    const size_t first = chunks.size();
    chunks.reserve(chunks.size() + ptrs.size());
    for (char * p : ptrs) {
        chunks.push_back(map_chunk(p, chunk_size, seed));
    }
    const auto t_mapped = clk::now();
    wait_mapped();
    const auto t_synced = clk::now();
    for (size_t i = first; i < chunks.size(); ++i) {
        k_mark<<<256, 256>>>(chunks[i].ptr, chunk_size/PAGE, seed);
    }
    const auto t_access = clk::now();
    // the launch time of the first kernel is counted with the rest: the bound is conservative
    const double gap   = std::chrono::duration<double, std::milli>(t_access - t_mapped).count();
    const double syncd = std::chrono::duration<double, std::milli>(t_synced - t_mapped).count();
    gap_max   = std::max(gap_max, gap);
    sync_max  = std::max(sync_max, syncd);
    other_max = std::max(other_max, gap - syncd);
    CHECK(hipGetLastError());
    check_all(chunks, "after a growth");
}

// grow a contiguous range from offset to total, batch chunks per step
static void grow(std::vector<chunk> & chunks, char * base, size_t & offset, size_t total, size_t chunk_size, int batch, uint32_t seed) {
    std::vector<char *> ptrs;
    while (offset < total) {
        ptrs.clear();
        for (int b = 0; b < batch && offset < total; ++b) {
            ptrs.push_back(base + offset);
            offset += chunk_size;
        }
        map_batch(chunks, ptrs, chunk_size, seed++);
    }
}

static void * reserve(size_t size) {
    void * base = nullptr;
    CHECK(hipMemAddressReserve(&base, size, align_, nullptr, 0));
    return base;
}

static void unmap_all(std::vector<chunk> & chunks) {
    CHECK(hipDeviceSynchronize());
    for (const auto & c : chunks) {
        unmap_chunk(c);
    }
    chunks.clear();
}

int main(int argc, char ** argv) {
    if (argc < 2 || strlen(argv[1]) != 1 || argv[1][0] < 'a' || argv[1][0] > 'd') {
        printf("usage: %s <a|b|c|d> [--sync none|wait|barrier|both] [--wait-ms N] [--chunk-mib N] [--batch N] [--target-gib N] [--barrier-mib N]\n", argv[0]);
        return 2;
    }
    const char cs = argv[1][0];
    size_t chunk_mib = 64;
    int    batch     = 4;
    size_t target    = 6ull << 30;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--sync") { sync_mode = argv[i + 1]; }
        else if (a == "--wait-ms") { wait_ms = atoi(argv[i + 1]); }
        else if (a == "--chunk-mib") { chunk_mib = (size_t) atoi(argv[i + 1]); }
        else if (a == "--batch") { batch = std::max(1, atoi(argv[i + 1])); }
        else if (a == "--target-gib") { target = (size_t) atoi(argv[i + 1]) << 30; }
        else if (a == "--barrier-mib") { barrier_size = (size_t) atoi(argv[i + 1])*MiB; }
        else { printf("unknown option %s\n", a.c_str()); return 2; }
    }
    CHECK(hipSetDevice(dev));
    hipMemAllocationProp prop = {};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = dev;
    size_t gran_rec = 0;
    CHECK(hipMemGetAllocationGranularity(&gran_rec, &prop, hipMemAllocationGranularityRecommended));
    align_ = std::max<size_t>(align_, gran_rec);
    const size_t chunk_size = chunk_mib*MiB;
    CHECK(hipMalloc(&bad_counter, sizeof(*bad_counter)));
    // load the kernels before the first mapping
    k_mark<<<1, 1>>>((char *) bad_counter, 0, 0);
    k_check<<<1, 1>>>((const char *) bad_counter, 0, 0, bad_counter);
    CHECK(hipDeviceSynchronize());

    size_t free_b = 0, total_b = 0;
    CHECK(hipMemGetInfo(&free_b, &total_b));
    target = std::min<size_t>(target, free_b > (1ull << 30) ? free_b - (1ull << 30) : free_b/2);
    target -= target % (chunk_size*batch);
    printf("case %c: sync %s, wait %d ms, barrier %zu MiB, chunk %zu MiB, batch %d, target %zu MiB (granularity %zu)\n",
        cs, sync_mode.c_str(), wait_ms, barrier_size/MiB, chunk_mib, batch, target/MiB, gran_rec);
    report_vram("at start");
    const size_t vram0 = vram_used();
    const auto t_start = clk::now();

    std::vector<chunk> chunks;
    size_t offset = 0;
    if (cs == 'a') {
        const size_t range = 9ull << 30;
        char * base = (char *) reserve(range);
        grow(chunks, base, offset, target, chunk_size, batch, 0x1000u);
        report_vram("grown");
        unmap_all(chunks);
        CHECK(hipMemAddressFree(base, range));
    } else if (cs == 'b') {
        const size_t range = 9ull << 30;
        const size_t full  = std::min<size_t>(target, 3ull << 30) / chunk_size * chunk_size;
        char * base = (char *) reserve(range);
        grow(chunks, base, offset, full, chunk_size, batch, 0x2000u);
        report_vram("grown");
        // shrink: unmap the upper half, the lower half keeps its data
        const size_t keep = chunks.size()/2;
        CHECK(hipDeviceSynchronize());
        for (size_t i = keep; i < chunks.size(); ++i) {
            unmap_chunk(chunks[i]);
        }
        chunks.resize(keep);
        report_vram("shrunk (same address range)");
        check_all(chunks, "the kept half after the shrink");
        // grow again on the same addresses
        offset = keep*chunk_size;
        grow(chunks, base, offset, full, chunk_size, batch, 0x3000u);
        report_vram("grown again on the same addresses");
        const size_t vram_grown = vram_used();
        unmap_all(chunks);
        CHECK(hipMemAddressFree(base, range));
        const int ms = vram0 ? wait_vram_below(vram_grown - (vram_grown - std::min(vram_grown, vram0))/2) : 0;
        if (ms < 0) {
            report_vram("whole range freed");
            printf("FAIL: the VRAM of the freed range did not come back within 3 s\n");
            return 1;
        }
        printf("  the VRAM of the freed range came back in %d ms\n", ms);
        // the way the KV shrinks: a new address range
        char * base2 = (char *) reserve(range);
        offset = 0;
        grow(chunks, base2, offset, full/2 - (full/2) % chunk_size, chunk_size, batch, 0x4000u);
        report_vram("new range grown");
        unmap_all(chunks);
        CHECK(hipMemAddressFree(base2, range));
    } else if (cs == 'c') {
        // fill the VRAM with hipMalloc, leaving room for the growth and a margin
        const size_t grow_size = std::min<size_t>(1ull << 30, target) / chunk_size * chunk_size;
        std::vector<void *> filler;
        size_t filled = 0;
        for (;;) {
            CHECK(hipMemGetInfo(&free_b, &total_b));
            if (free_b < grow_size + 512*MiB + 256*MiB) {
                break;
            }
            void * p = nullptr;
            if (hipMalloc(&p, 256*MiB) != hipSuccess) {
                (void) hipGetLastError();
                break;
            }
            CHECK(hipMemset(p, 1, 256*MiB));
            filler.push_back(p);
            filled += 256*MiB;
        }
        CHECK(hipDeviceSynchronize());
        printf("  filler: %zu MiB with hipMalloc\n", filled/MiB);
        report_vram("after the filler");
        const size_t range = 9ull << 30;
        char * base = (char *) reserve(range);
        grow(chunks, base, offset, grow_size, chunk_size, batch, 0x5000u);
        report_vram("grown");
        unmap_all(chunks);
        CHECK(hipMemAddressFree(base, range));
        for (void * p : filler) {
            CHECK(hipFree(p));
        }
    } else {
        // separate address ranges, batch of them per step
        const size_t per_size = 32*MiB;
        const size_t chunk_d  = std::min<size_t>(chunk_size, per_size);
        std::vector<char *> ranges;
        for (int s = 0; s < 96; ++s) {
            ranges.push_back((char *) reserve(per_size));
        }
        std::vector<char *> ptrs;
        for (size_t s = 0; s < ranges.size(); s += batch) {
            ptrs.assign(ranges.begin() + s, ranges.begin() + std::min(ranges.size(), s + batch));
            map_batch(chunks, ptrs, chunk_d, 0x6000u + (uint32_t) s);
        }
        report_vram("grown");
        unmap_all(chunks);
        for (char * r : ranges) {
            CHECK(hipMemAddressFree(r, per_size));
        }
    }
    const double run_s = std::chrono::duration<double>(clk::now() - t_start).count();
    printf("  mapping end -> first access: max %.2f ms, of which the sync %.2f ms, anything else at most %.3f ms\n", gap_max, sync_max, other_max);
    CHECK(hipDeviceSynchronize());
    const int ms = vram0 ? wait_vram_below(vram0 + 128*MiB) : 0;
    report_vram("after freeing everything");
    if (ms < 0) {
        printf("FAIL: the VRAM did not come back within 3 s (%zu -> %zu MiB)\n", vram0/MiB, vram_used()/MiB);
        return 1;
    }
    printf("  freed: the VRAM came back in %d ms; run time %.1f s\n", ms, run_s);
    printf("case %c: OK\n", cs);
    return 0;
}
