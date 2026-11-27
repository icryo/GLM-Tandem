// CPU side of the in-encoder exchange prototype: a native thread that waits for the GPU's "partial
// ready" counter and answers with the "peer data arrived" counter (in the real design it would post
// the RDMA send of the window here and wait for the peer's receive). Bounded spins only.
#include <pthread.h>
#include <pthread/qos.h>
#include <stdint.h>
#include <mach/mach_time.h>
typedef struct { volatile uint32_t *g, *c; int n; long timeouts; uint64_t lat_sum; uint64_t lat_n; } P;
static P P0; static pthread_t T;
static void *run(void *a) {
  P *p = (P *)a;
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  for (int i = 1; i <= p->n; i++) {
    uint64_t t0 = mach_absolute_time(); long k = 0;
    while (__atomic_load_n(p->g, __ATOMIC_ACQUIRE) < (uint32_t)i) {
      if (++k > 400000000L) { p->timeouts++; return 0; }
    }
    __atomic_store_n(p->c, (uint32_t)i, __ATOMIC_RELEASE);
    (void)t0;
  }
  return 0;
}
int poller_start(void *g, void *c, int n) { P0.g = g; P0.c = c; P0.n = n; P0.timeouts = 0; return pthread_create(&T, 0, run, &P0); }
long poller_join(void) { pthread_join(T, 0); return P0.timeouts; }
