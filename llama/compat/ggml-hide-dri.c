// Dual-buffer fork (multimodel): an LD_PRELOAD library for the llama-server processes that load the ROCm backend.
//
// The ROCm runtime opens the render node of every GPU it finds in the KFD topology when it starts, and it fails as a
// whole on a card it cannot use (another backend's AMD card, an unsupported one), before ROCR_VISIBLE_DEVICES filters
// anything. While GGML_HIDE_RENDER_NODES lists render nodes (e.g. "renderD129,renderD130"), opening, accessing or
// them fails with ENOENT, as if they were not in the container. ggml sets the variable only while the HIP
// backend loads (GGML_HIP_HIDE_RENDER_NODES, ggml-backend-reg.cpp), so the other backends of the same process (Vulkan)
// still see their cards. Without the variable every call goes straight to libc.

// the fortified open() is an inline wrapper: this file defines the real symbols
#undef _FORTIFY_SOURCE
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// path is one of the hidden render nodes: /dev/dri/renderD<N> (or a path ending in it) whose name is in the list
static int hidden(const char * path) {
    const char * list = getenv("GGML_HIDE_RENDER_NODES");
    if (!list || !*list || !path) {
        return 0;
    }
    const char * name = strrchr(path, '/');
    name = name ? name + 1 : path;
    if (strncmp(name, "renderD", 7) != 0 || strstr(path, "/dri/") == NULL) {
        return 0;
    }
    const size_t n = strlen(name);
    for (const char * p = list; *p;) {
        const char * end = strchr(p, ',');
        const size_t len = end ? (size_t) (end - p) : strlen(p);
        if (len == n && strncmp(p, name, n) == 0) {
            return 1;
        }
        if (!end) {
            break;
        }
        p = end + 1;
    }
    return 0;
}

#define REAL(name, ret, ...) \
    static ret (*real_##name)(__VA_ARGS__) = NULL; \
    if (!real_##name) { real_##name = (ret (*)(__VA_ARGS__)) dlsym(RTLD_NEXT, #name); } \n    if (!real_##name) { errno = ENOSYS; return -1; }

static mode_t open_mode(int flags, va_list ap) {
#ifdef O_TMPFILE
    if ((flags & O_TMPFILE) == O_TMPFILE) {
        return (mode_t) va_arg(ap, int);
    }
#endif
    return (flags & O_CREAT) ? (mode_t) va_arg(ap, int) : 0;
}

int open(const char * path, int flags, ...) {
    va_list ap; va_start(ap, flags); mode_t mode = open_mode(flags, ap); va_end(ap);
    if (hidden(path)) { errno = ENOENT; return -1; }
    REAL(open, int, const char *, int, ...)
    return real_open(path, flags, mode);
}

int open64(const char * path, int flags, ...) {
    va_list ap; va_start(ap, flags); mode_t mode = open_mode(flags, ap); va_end(ap);
    if (hidden(path)) { errno = ENOENT; return -1; }
    REAL(open64, int, const char *, int, ...)
    return real_open64(path, flags, mode);
}

int openat(int dirfd, const char * path, int flags, ...) {
    va_list ap; va_start(ap, flags); mode_t mode = open_mode(flags, ap); va_end(ap);
    if (hidden(path)) { errno = ENOENT; return -1; }
    REAL(openat, int, int, const char *, int, ...)
    return real_openat(dirfd, path, flags, mode);
}

int openat64(int dirfd, const char * path, int flags, ...) {
    va_list ap; va_start(ap, flags); mode_t mode = open_mode(flags, ap); va_end(ap);
    if (hidden(path)) { errno = ENOENT; return -1; }
    REAL(openat64, int, int, const char *, int, ...)
    return real_openat64(dirfd, path, flags, mode);
}

int __open_2(const char * path, int flags) {
    if (hidden(path)) { errno = ENOENT; return -1; }
    REAL(__open_2, int, const char *, int)
    return real___open_2(path, flags);
}

int __open64_2(const char * path, int flags) {
    if (hidden(path)) { errno = ENOENT; return -1; }
    REAL(__open64_2, int, const char *, int)
    return real___open64_2(path, flags);
}

int access(const char * path, int mode) {
    if (hidden(path)) { errno = ENOENT; return -1; }
    REAL(access, int, const char *, int)
    return real_access(path, mode);
}

int faccessat(int dirfd, const char * path, int mode, int flags) {
    if (hidden(path)) { errno = ENOENT; return -1; }
    REAL(faccessat, int, int, const char *, int, int)
    return real_faccessat(dirfd, path, mode, flags);
}
