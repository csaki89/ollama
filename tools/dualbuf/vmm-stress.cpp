// VMM stress test for ROCm (HIP, one source file). A GPU page fault kills the process, so every run is a separate process (vmm-stress.sh).
//   vmm-stress <case> [--sync none|wait|barrier|both] [--wait-ms N] [--chunk-mib N] [--batch N] [--target-gib N] [--barrier-mib N]
// Cases (every chunk is completely filled by a kernel and read back by another one):
//   a  one 9 GiB address range, growth up to --target-gib with chunks of --chunk-mib
//   b  growth, shrink (unmap the upper half), growth again on the same addresses; then free the whole range and start in a new one
//   c  the VRAM almost filled with hipMalloc first, then growth
//   d  many separate address ranges one after the other (96 x 32 MiB, one chunk each)
// Sync after mapping and before the first access:
//   none     nothing
//   wait     sleep --wait-ms
//   barrier  map a fresh allocation of --barrier-mib through KFD (hipMalloc + hipFree), which waits for the page table updates queued before it
//   both     barrier, then wait
// --batch N: map N chunks, sync once, then access them (like the KV growth of llama.cpp)
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

static const size_t MiB = 1ull << 20;

__global__ void k_fill(uint32_t * p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x*(size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) {
        p[i] = (uint32_t) i ^ seed;
    }
}

__global__ void k_check(const uint32_t * p, size_t n, uint32_t seed, unsigned long long * bad) {
    for (size_t i = blockIdx.x*(size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) {
        if (p[i] != ((uint32_t) i ^ seed)) {
            atomicAdd(bad, 1ull);
        }
    }
}

struct chunk {
    char * ptr;
    size_t size;
    hipMemGenericAllocationHandle_t handle;
};

static std::string sync_mode = "both";
static int    wait_ms     = 5;
static size_t barrier_size = 4*MiB;
static size_t align_      = 2*MiB;
static int    dev         = 0;
static unsigned long long * bad_counter = nullptr;

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
    printf("  VRAM %-34s used(sysfs) %6zu MiB, hipMemGetInfo free %6zu of %6zu MiB\n", what, vram_used()/MiB, free_b/MiB, total_b/MiB);
    fflush(stdout);
}

static void wait_mapped() {
    if (sync_mode == "barrier" || sync_mode == "both") {
        void * p = nullptr;
        const hipError_t e = hipMalloc(&p, barrier_size);
        if (e == hipSuccess) {
            CHECK(hipFree(p));
        } else {
            printf("  barrier allocation failed: %s\n", hipGetErrorString(e));
        }
    }
    if (sync_mode == "wait" || sync_mode == "both") {
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    }
}

static chunk map_chunk(char * ptr, size_t size) {
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
    return { ptr, size, h };
}

static void unmap_chunk(const chunk & c) {
    CHECK(hipMemUnmap(c.ptr, c.size));
    CHECK(hipMemRelease(c.handle));
}

// fill completely with a kernel, read back with another one
static void fill_check(const chunk & c, uint32_t seed) {
    const size_t n = c.size/4;
    k_fill<<<1024, 256>>>((uint32_t *) c.ptr, n, seed);
    CHECK(hipGetLastError());
    CHECK(hipMemset(bad_counter, 0, sizeof(*bad_counter)));
    k_check<<<1024, 256>>>((const uint32_t *) c.ptr, n, seed, bad_counter);
    CHECK(hipDeviceSynchronize());
    unsigned long long bad = 0;
    CHECK(hipMemcpy(&bad, bad_counter, sizeof(bad), hipMemcpyDeviceToHost));
    if (bad != 0) {
        printf("FAIL: %llu wrong words in %p + %zu\n", bad, (void *) c.ptr, c.size);
        fflush(stdout);
        exit(1);
    }
}

static void check_only(const chunk & c, uint32_t seed) {
    CHECK(hipMemset(bad_counter, 0, sizeof(*bad_counter)));
    k_check<<<1024, 256>>>((const uint32_t *) c.ptr, c.size/4, seed, bad_counter);
    CHECK(hipDeviceSynchronize());
    unsigned long long bad = 0;
    CHECK(hipMemcpy(&bad, bad_counter, sizeof(bad), hipMemcpyDeviceToHost));
    if (bad != 0) {
        printf("FAIL: %llu wrong words in the kept chunk %p + %zu\n", bad, (void *) c.ptr, c.size);
        fflush(stdout);
        exit(1);
    }
}

// map chunks at base + offset up to total bytes, in batches: map, sync once, then the first access of every chunk
static void grow(std::vector<chunk> & chunks, char * base, size_t & offset, size_t total, size_t chunk_size, int batch, uint32_t seed) {
    while (offset < total) {
        const size_t first = chunks.size();
        for (int b = 0; b < batch && offset < total; ++b) {
            chunks.push_back(map_chunk(base + offset, chunk_size));
            offset += chunk_size;
        }
        wait_mapped();
        for (size_t i = first; i < chunks.size(); ++i) {
            fill_check(chunks[i], seed);
        }
    }
}

static void * reserve(size_t size) {
    void * base = nullptr;
    CHECK(hipMemAddressReserve(&base, size, align_, nullptr, 0));
    return base;
}

int main(int argc, char ** argv) {
    if (argc < 2 || strlen(argv[1]) != 1 || argv[1][0] < 'a' || argv[1][0] > 'd') {
        printf("usage: %s <a|b|c|d> [--sync none|wait|barrier|both] [--wait-ms N] [--chunk-mib N] [--batch N] [--target-gib N] [--barrier-mib N]\n", argv[0]);
        return 2;
    }
    const char cs = argv[1][0];
    size_t chunk_mib = 64;
    int    batch     = 1;
    size_t target    = 6ull << 30;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--sync") { sync_mode = argv[i + 1]; }
        else if (a == "--wait-ms") { wait_ms = atoi(argv[i + 1]); }
        else if (a == "--chunk-mib") { chunk_mib = atoi(argv[i + 1]); }
        else if (a == "--batch") { batch = atoi(argv[i + 1]); }
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
    align_ = std::max(align_, gran_rec);
    const size_t chunk_size = chunk_mib*MiB;
    CHECK(hipMalloc(&bad_counter, sizeof(*bad_counter)));

    size_t free_b = 0, total_b = 0;
    CHECK(hipMemGetInfo(&free_b, &total_b));
    target = std::min<size_t>(target, free_b > (1ull << 30) ? free_b - (1ull << 30) : free_b/2);
    target -= target % chunk_size;
    printf("case %c: sync %s, wait %d ms, barrier %zu MiB, chunk %zu MiB, batch %d, target %zu MiB (granularity %zu)\n",
        cs, sync_mode.c_str(), wait_ms, barrier_size/MiB, chunk_mib, batch, target/MiB, gran_rec);
    report_vram("at start");
    const size_t vram0 = vram_used();

    std::vector<chunk> chunks;
    size_t offset = 0;
    if (cs == 'a') {
        const size_t range = 9ull << 30;
        char * base = (char *) reserve(range);
        grow(chunks, base, offset, target, chunk_size, batch, 0x1111u);
        report_vram("grown");
        for (const auto & c : chunks) {
            check_only(c, 0x1111u);
        }
        CHECK(hipDeviceSynchronize());
        for (const auto & c : chunks) {
            unmap_chunk(c);
        }
        CHECK(hipMemAddressFree(base, range));
    } else if (cs == 'b') {
        const size_t range = 9ull << 30;
        const size_t full  = std::min<size_t>(target, 3ull << 30) / chunk_size * chunk_size;
        char * base = (char *) reserve(range);
        grow(chunks, base, offset, full, chunk_size, batch, 0x2222u);
        report_vram("grown");
        // shrink: unmap the upper half, the lower half keeps its data
        const size_t keep = chunks.size()/2;
        CHECK(hipDeviceSynchronize());
        for (size_t i = keep; i < chunks.size(); ++i) {
            unmap_chunk(chunks[i]);
        }
        chunks.resize(keep);
        report_vram("shrunk (same address range)");
        for (const auto & c : chunks) {
            check_only(c, 0x2222u);
        }
        // grow again on the same addresses
        offset = keep*chunk_size;
        const size_t first_new = chunks.size();
        grow(chunks, base, offset, full, chunk_size, batch, 0x3333u);
        report_vram("grown again on the same addresses");
        for (size_t i = 0; i < chunks.size(); ++i) {
            check_only(chunks[i], i < first_new ? 0x2222u : 0x3333u);
        }
        CHECK(hipDeviceSynchronize());
        for (const auto & c : chunks) {
            unmap_chunk(c);
        }
        chunks.clear();
        CHECK(hipMemAddressFree(base, range));
        report_vram("whole range freed");
        // the way the KV shrinks: a new address range
        char * base2 = (char *) reserve(range);
        offset = 0;
        grow(chunks, base2, offset, full/2 - (full/2) % chunk_size, chunk_size, batch, 0x4444u);
        report_vram("new range grown");
        CHECK(hipDeviceSynchronize());
        for (const auto & c : chunks) {
            unmap_chunk(c);
        }
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
        grow(chunks, base, offset, grow_size, chunk_size, batch, 0x5555u);
        report_vram("grown");
        CHECK(hipDeviceSynchronize());
        for (const auto & c : chunks) {
            unmap_chunk(c);
        }
        CHECK(hipMemAddressFree(base, range));
        for (void * p : filler) {
            CHECK(hipFree(p));
        }
    } else {
        // many separate address ranges, one after the other
        const size_t per_size = 32*MiB;
        const size_t chunk_d  = std::min(chunk_size, per_size);
        std::vector<std::pair<char *, size_t>> ranges;
        for (int s = 0; s < 96; ++s) {
            char * base = (char *) reserve(per_size);
            ranges.push_back({ base, per_size });
            chunks.push_back(map_chunk(base, chunk_d));
            if (batch <= 1 || (s + 1) % batch == 0 || s == 95) {
                wait_mapped();
                for (size_t i = chunks.size() - (batch <= 1 ? 1 : ((size_t) s % batch) + 1); i < chunks.size(); ++i) {
                    fill_check(chunks[i], 0x6666u + (uint32_t) i);
                }
            }
        }
        report_vram("grown");
        for (size_t i = 0; i < chunks.size(); ++i) {
            check_only(chunks[i], 0x6666u + (uint32_t) i);
        }
        CHECK(hipDeviceSynchronize());
        for (const auto & c : chunks) {
            unmap_chunk(c);
        }
        for (const auto & r : ranges) {
            CHECK(hipMemAddressFree(r.first, r.second));
        }
    }
    CHECK(hipDeviceSynchronize());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    report_vram("after freeing everything");
    const size_t vram1 = vram_used();
    if (vram0 && vram1 > vram0 + 128*MiB) {
        printf("FAIL: the VRAM did not come back (%zu -> %zu MiB)\n", vram0/MiB, vram1/MiB);
        return 1;
    }
    printf("case %c: OK\n", cs);
    return 0;
}
