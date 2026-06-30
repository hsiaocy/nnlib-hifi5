#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xa_nnlib_common.h"
#include "xa_nn_conv2d_std_state.h"

#include "xtensa/tie/xt_core.h"
#include "xtensa/xtruntime.h"

/* Maximum file size ~1 MB (covers all patterns including p06 many_channels) */
#define MAX_FILE_BYTES  (1024 * 1024)




int main(void) {
    unsigned int c0, c1;
    
    c0 = XT_RSR_CCOUNT();
    c1 = XT_RSR_CCOUNT();
    printf("empty: %u\n", c1 - c0);
    
    c0 = XT_RSR_CCOUNT();
    volatile int sum = 0;
    for (int i = 0; i < 1000; i++) {
        sum += i;
    }
    c1 = XT_RSR_CCOUNT();
    printf("1000-iter loop: %u\n", c1 - c0);
    return 0;
}

