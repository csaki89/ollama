// VMM reproducer for the dynamic KV of the dual-buffer branch (HIP, one source file).
// Compares the VMM layouts of the growable KV buffers; a GPU page fault kills the process, so every case runs on its own:
//   vmm-repro a   old layout: 96 address ranges of 32 MiB, one 4 MiB chunk each (recommended granularity), async zeroing on an own stream
//   vmm-repro b   new layout: one address range for all 96 sections of 32 MiB, chunks aligned to max(recommended, 2 MiB),
//                 synchronous zeroing right after each mapping, then growth by one more chunk per section
//   vmm-repro c   old layout with synchronous zeroing (hipMemset + hipDeviceSynchronize)
//   vmm-repro d   old layout with the reservation and chunk alignment max(recommended, 2 MiB)
// Every case then writes a pattern with a kernel and checks it with a second kernel.
// Options: vmm-repro <case> [sections=96] [section_mib=32] [chunk_mib=4]
// Build: hipcc -O2 --offload-arch=gfx1030 -o vmm-repro vmm-repro.cpp  (run with HSA_OVERRIDE_GFX_VERSION=10.3.0 on gfx1031)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("FAIL %s: %s (%d) at line %d\n", #x, hipGetErrorString(e_), (int) e_, __LINE__); exit(1); } } while (0)

__global__ void k_fill(uint32_t * p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x*(size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) {
        p[i] = (uint32_t) i ^ seed;
    }
}

__global__ void k_check(const uint32_t * p, size_t n, uint32_t seed, bool zero, unsigned long long * bad) {
    for (size_t i = blockIdx.x*(size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x*blockDim.x) {
        const uint32_t want = zero ? 0u : ((uint32_t) i ^ seed);
        if (p[i] != want) {
            atomicAdd(bad, 1ull);
        }
    }
}

struct region {
    char * ptr;      // start of the mapped part
    size_t size;     // mapped bytes
};

static int dev = 0;
static size_t gran_rec = 0, gran_min = 0;

static hipMemAllocationProp make_prop() {
    hipMemAllocationProp prop = {};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = dev;
    return prop;
}

static void map_chunk(char * ptr, size_t size, std::vector<hipMemGenericAllocationHandle_t> & handles) {
    hipMemAllocationProp prop = make_prop();
    hipMemGenericAllocationHandle_t h;
    CHECK(hipMemCreate(&h, size, &prop, 0));
    CHECK(hipMemMap(ptr, size, 0, h, 0));
    hipMemAccessDesc access = {};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id   = dev;
    access.flags         = hipMemAccessFlagsProtReadWrite;
    CHECK(hipMemSetAccess(ptr, size, &access, 1));
    handles.push_back(h);
}

static size_t round_up(size_t x, size_t a) {
    return a*((x + a - 1)/a);
}

static void check_regions(const std::vector<region> & regs, uint32_t seed) {
    unsigned long long * bad = nullptr;
    CHECK(hipMalloc(&bad, sizeof(*bad)));
    for (const auto & r : regs) {
        const size_t n = r.size/4;
        CHECK(hipMemset(bad, 0, sizeof(*bad)));
        k_check<<<256, 256>>>((const uint32_t *) r.ptr, n, seed, true, bad);
        k_fill <<<256, 256>>>((uint32_t *) r.ptr, n, seed);
        k_check<<<256, 256>>>((const uint32_t *) r.ptr, n, seed, false, bad);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        unsigned long long h_bad = 0;
        CHECK(hipMemcpy(&h_bad, bad, sizeof(h_bad), hipMemcpyDeviceToHost));
        if (h_bad != 0) {
            printf("FAIL: %llu wrong words in %p + %zu\n", h_bad, (void *) r.ptr, r.size);
            exit(1);
        }
    }
    CHECK(hipFree(bad));
}

int main(int argc, char ** argv) {
    if (argc < 2 || strlen(argv[1]) != 1 || argv[1][0] < 'a' || argv[1][0] > 'd') {
        printf("usage: %s <a|b|c|d> [sections=96] [section_mib=32] [chunk_mib=4]\n", argv[0]);
        return 2;
    }
    const char   cs        = argv[1][0];
    const int    n_sec     = argc > 2 ? atoi(argv[2]) : 96;
    const size_t sec_size  = (size_t) (argc > 3 ? atoi(argv[3]) : 32) << 20;
    const size_t chunk_req = (size_t) (argc > 4 ? atoi(argv[4]) : 4)  << 20;

    CHECK(hipSetDevice(dev));
    int vmm = 0;
    CHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, dev));
    hipMemAllocationProp prop = make_prop();
    CHECK(hipMemGetAllocationGranularity(&gran_rec, &prop, hipMemAllocationGranularityRecommended));
    CHECK(hipMemGetAllocationGranularity(&gran_min, &prop, hipMemAllocationGranularityMinimum));
    const size_t align2m = round_up(std::max<size_t>(gran_rec, 2u << 20), gran_min ? gran_min : 1);
    printf("case %c: VMM %d, granularity recommended %zu, minimum %zu; %d sections of %zu MiB, chunk %zu MiB\n",
        cs, vmm, gran_rec, gran_min, n_sec, sec_size >> 20, chunk_req >> 20);

    std::vector<region> regs;
    std::vector<hipMemGenericAllocationHandle_t> handles;
    std::vector<std::pair<void *, size_t>> ranges;

    if (cs == 'b') {
        // one address range, a section per tensor, chunks aligned to max(recommended, 2 MiB), synchronous zeroing after each mapping
        const size_t sec = round_up(sec_size, align2m);
        const size_t total = sec*n_sec;
        void * base = nullptr;
        CHECK(hipMemAddressReserve(&base, total, align2m, nullptr, 0));
        ranges.push_back({ base, total });
        const size_t chunk = round_up(chunk_req, align2m);
        printf("  one range %p, %zu MiB, section %zu MiB, chunk %zu MiB (alignment %zu)\n", base, total >> 20, sec >> 20, chunk >> 20, align2m);
        for (int s = 0; s < n_sec; ++s) {
            char * p = (char *) base + s*sec;
            map_chunk(p, chunk, handles);
            CHECK(hipMemset(p, 0, chunk));
            CHECK(hipDeviceSynchronize());
            regs.push_back({ p, chunk });
        }
        check_regions(regs, 0x1234u);
        // growth: one more chunk per section, in place
        for (int s = 0; s < n_sec; ++s) {
            char * p = (char *) base + s*sec;
            if (2*chunk <= sec) {
                map_chunk(p + chunk, chunk, handles);
                CHECK(hipMemset(p + chunk, 0, chunk));
                CHECK(hipDeviceSynchronize());
                regs[s].size = 2*chunk;
            }
        }
        // the first half holds the pattern, zero it before the combined check
        for (auto & r : regs) {
            CHECK(hipMemset(r.ptr, 0, r.size));
        }
        CHECK(hipDeviceSynchronize());
        check_regions(regs, 0x5678u);
    } else {
        // old layout: one address range per tensor
        const size_t align = cs == 'd' ? align2m : gran_rec;
        const size_t sec   = round_up(sec_size, align);
        const size_t chunk = round_up(chunk_req, align);
        hipStream_t stream = nullptr;
        if (cs == 'a') {
            CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        }
        printf("  %d ranges of %zu MiB, chunk %zu bytes, reservation alignment %zu, zeroing %s\n", n_sec, sec >> 20, chunk,
            cs == 'd' ? align : (size_t) 0, cs == 'a' ? "async on an own stream" : "synchronous");
        for (int s = 0; s < n_sec; ++s) {
            void * base = nullptr;
            CHECK(hipMemAddressReserve(&base, sec, cs == 'd' ? align : 0, nullptr, 0));
            ranges.push_back({ base, sec });
            map_chunk((char *) base, chunk, handles);
            regs.push_back({ (char *) base, chunk });
        }
        for (const auto & r : regs) {
            if (cs == 'a') {
                CHECK(hipMemsetAsync(r.ptr, 0, r.size, stream));
            } else {
                CHECK(hipMemset(r.ptr, 0, r.size));
            }
        }
        if (cs == 'a') {
            CHECK(hipStreamSynchronize(stream));
            CHECK(hipStreamDestroy(stream));
        } else {
            CHECK(hipDeviceSynchronize());
        }
        check_regions(regs, 0x1234u);
    }

    // release: unmap every chunk, release its handle, then free the address ranges
    CHECK(hipDeviceSynchronize());
    size_t ih = 0;
    for (const auto & r : regs) {
        size_t chunk = r.size;
        if (cs == 'b' && handles.size() == 2*regs.size()) {
            chunk = r.size/2;
            CHECK(hipMemUnmap(r.ptr, chunk));
            CHECK(hipMemUnmap(r.ptr + chunk, chunk));
            CHECK(hipMemRelease(handles[ih++]));
        } else {
            CHECK(hipMemUnmap(r.ptr, chunk));
        }
    }
    for (; ih < handles.size(); ++ih) {
        CHECK(hipMemRelease(handles[ih]));
    }
    for (const auto & rg : ranges) {
        CHECK(hipMemAddressFree(rg.first, rg.second));
    }
    printf("case %c: OK\n", cs);
    return 0;
}
