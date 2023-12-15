#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define Max(a,b) ((a)>(b)?(a):(b))

#define N (1000)
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
	init(A);
	for(it=1; it<=itmax; it++)
	{
		double eps = relax(A,B);
		resid(A,B);
		printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}
	verify(A);
	double end = omp_get_wtime();
	printf( "time = %3.10f seconds.\n", end - start);
	return 0;
}

void init(float A [N][N])
{ 
	for(int i=0; i<=N-1; i++)
	for(int j=0; j<=N-1; j++)
	
	{
		if(i==0 || i==N-1 || j==0 || j==N-1) A[i][j]= 0.;
		else A[i][j]= ( 1. + i + j ) ;
	}
} 

double relax(float A [N][N], float  B [N][N])
{
	double eps = 0.;
	for(int i=1; i<=N-2; i++)
	for(int j=1; j<=N-2; j++)
	{
		B[i][j]=(A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4.;
		eps = Max( eps, fabs(A[i][j] - B[i][j]) );
	}

	return eps;
}

void resid(float A [N][N], float  B [N][N])
{ 
	double eps = 0.;
	for(int i=1; i<=N-2; i++)
	for(int j=1; j<=N-2; j++)

	{
		float e;
		e = fabs(A[i][j] - B[i][j]);         
		A[i][j] = B[i][j]; 
		eps = Max(eps,e);
	}
}

void verify(float A [N][N])
{
	double s = 0.;
	for(int i=0; i<=N-1; i++)
	for(int j=0; j<=N-1; j++)
	
	{
		s=s+A[i][j]*(i+1)*(j+1)/(N*N);
	}
	printf("  S = %f\n",s);
}
