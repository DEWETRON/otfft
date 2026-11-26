#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc,char**argv){
  fftw_init_threads();
  int nth=atoi(argv[1]);
  fftw_plan_with_nthreads(nth);
  fftw_set_timelimit(10.0);
  for(int i=2;i<argc;i++){
    int n=atoi(argv[i]);
    fftw_complex*in=fftw_malloc(sizeof(fftw_complex)*n),*out=fftw_malloc(sizeof(fftw_complex)*n);
    fftw_plan p=fftw_plan_dft_1d(n,in,out,FFTW_FORWARD,FFTW_MEASURE);
    printf("=== N=%d threads=%d\n",n,nth); fftw_print_plan(p); printf("\n");
    double add,mul,fma; fftw_flops(p,&add,&mul,&fma); printf("flops add %.0f mul %.0f fma %.0f\n",add,mul,fma);
    fftw_destroy_plan(p); fftw_free(in); fftw_free(out);
  }
}
