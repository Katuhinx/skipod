#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define N (1000)
#define THR (10)
#define TASK 1

float maxeps = 0.1e-7;
int itmax = 100;
float A [N][N],  B [N][N];

double relax(float A [N][N], float  B [N][N]);
void resid(float A [N][N], float  B [N][N]);
void init(float A [N][N]);
void verify(float A [N][N]); 

int main(int an, char **as)
{
	int it;
	double start = omp_get_wtime();
    float (*TempA)[N] = A, (*TempB)[N] = B;
	init(TempA);
	for(it=1; it<=itmax; it++)
	{
		double eps = relax(TempA,TempB);
		resid(A,B);
		printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}
	verify(TempA);
	double end = omp_get_wtime();
	printf( "time = %3.10f seconds.\n", end - start);
	return 0;
}

void init(float A [N][N])
{ 
	int i, j;

	#pragma omp parallel shared(A,B) num_threads(THR) private(i,j) 
	{
		#pragma omp for
		for(i=0; i<=N-1; i++)
		for(j=0; j<=N-1; j++)
		{
			if(i==0 || i==N-1 || j==0 || j==N-1) A[i][j]= 0.;
			else A[i][j]= ( 1. + i + j ) ;
		}
	}
} 

double relax(float A [N][N], float  B [N][N])
{
	double eps = 0.;
	int i, j, n;
	#pragma omp parallel shared(A,B) num_threads(THR) private(i,j) reduction(max:eps)
	{
		#pragma omp single 
        {
            for(i=1; i<=N-2; i+=TASK)
		    {
			    double e = 0.;
				#pragma omp task shared(A, B, eps)
                {
                    for(n=i; n<i+TASK; n++)
	                for(j=1; j<=N-2; j++)
                    {
		                B[n][j]=(A[n-1][j]+A[n+1][j]+A[n][j-1]+A[n][j+1])/4.;
                        e = Max( eps, fabs(A[n][j] - B[n][j]) );
	                }	

                    eps = Max(e,eps);	        
                }
		    }
        }
	}

	return eps;
}

void resid(float A [N][N], float  B [N][N])
{ 
	double eps = 0.;
	int i, j;
	#pragma omp parallel shared(A,B) num_threads(THR) private(i,j) reduction(max:eps)
	{
		#pragma omp for
		for(int i=1; i<=N-2; i++)
		for(int j=1; j<=N-2; j++)
		{
			float e;
			e = fabs(A[i][j] - B[i][j]);         
			A[i][j] = B[i][j]; 
			eps = Max(eps,e);
		}
	}
	
}

void verify(float A [N][N])
{
	double s = 0.;
	int i,j;
	#pragma omp parallel shared(A,B) num_threads(THR) private(i,j) reduction(+:s) 
	{
		#pragma omp for
		for(i=0; i<=N-1; i++)
		for(j=0; j<=N-1; j++)
	
		{
			s=s+A[i][j]*(i+1)*(j+1)/(N*N);
		}
	}

	printf("  S = %f\n",s);
}
