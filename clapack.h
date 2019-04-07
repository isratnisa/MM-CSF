
#ifndef __LAPACK_H
#define __LAPACK_H

#ifdef __cplusplus 	
extern "C" {	
#endif		

/* Cholesky factorization */
void spotrf_(
    char *,
    int *,
    float *,
    int *,
    int *);
void spotrs_(
    char *, int *,
    int *,
    float *,
    int *,
    float *,
    int *,
    int *);

/* Rank-k update. */
void ssyrk_(
    char *,
    char *,
    int *,
    int *,
    float *,
    float *,
    int *,
    float *,
    float *,
    int *);

/* LU */
void sgetrf_(
    int *,
    int *,
    float *,
    int *,
    int *,
    int *);
void sgetrs_(
    char *,
    int *,
    int *,
    float *,
    int *,
    int *,
    float *,
    int *,
    int *);
void sgesv_(
    int *,
    int *,
    float *,
    int *,
    int *,
    float *,
    int *,
    int *);





#ifdef __cplusplus
}
#endif


#endif /* __LAPACK_H */
