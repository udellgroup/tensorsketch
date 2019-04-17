/*
// TENSORSKETCHMATC3.C 	Computes the sketch of the transpose of input matrix. The sketch
//						is the TensorSketch corresponding to the input hashing functions
// 						h and s.
//
// INPUTS:
// 	M 		 	- The input matrix. Note that the transpose of the sketch of the transpose of M will be
// 				  returned. So if we give M as an input, we get back
//					(TensorSketch(M.')).'
// 	h 			- A row or column cell containing column vector representing the hashing
// 				  functions. Each column vector should be of type int32.
// 	s 			- A row or column cell containing column vectors representing the sign
// 				  hashing function. The column vectors should be left as double arrays.
// 	sketch_dim 	- The target sketch dimension. Should be of type int32.
//
// OPTIONAL INPUTS:
//	outer_dim_start 	- Use when sketching a large tensor in pieces; see Note 4 for more info.
//						  Needs to be int32. Give as MATLAB indexing, i.e. start at 1.
//	outer_dim_end		- Use when sketching a large tensor in pieces; see Note 4 for more info.
//						  Needs to be int32. Give as MATLAB indexing, i.e. start at 1.
//
// OUTPUTS:
// 	MsTT 		- The transpose of the TensorSketch of the transpose of M, i.e.
//					MsTT = (TensorSketch(M.')).'
//				  This somewhat convoluted return variable is due to optimization of the 
//				  memory access patterns.
//
// NOTES:
// 	Note 1 		- Note that compute_row() is called recursively to simulate the necessary
// 				  number of for loops needed to compute the target_row.
// 	Note 2 		- Note that in the inner-most for loop in compute_row(), the access pattern
// 				  to input_matrix is optimized by incrementing the index one step at a time.
//	Note 3 		- I made an additional change to the output_matrix so that this matrix also
//				  is accessed efficiently when updating it in the inner-most for loop in
//				  compute_row(). That change alone made this function twice as fast, even
//				  taking the additional time of doing an extra transpose in MATLAB into
//				  consideration.
//	Note 4		- When dealing e.g. with large dense tensors that we have to load piece-
//				  by-piece into RAM, we would like to do sketches of parts of tensors.
//				  When sketching a part of a tensor, the outermost loop in compute_row()
//				  will not go over all indices. The optional input variables outer_dim_start
//				  and outer_dim_end keep track of where this outermost loop starts and ends.
//				  They are given in terms of the indices, i.e. 
//					0 <= outer_dim_start <= outer_dim_end < I(N'),
//				  where N' is the largest dimension along which we don't matricize (so N' = N 
//				  always, except when we matricize along dimension N, in which case N' = N-1).
*/

/*
// Author:   Osman Asif Malik
// Email:    osman.malik@colorado.edu
// Date:     May 24, 2018
*/

#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Declare matrix global variables */
double *input_matrix;
const mwSize *dim_array;
mwSize no_rows;
mwSize no_cols;

/* Declare hashing and sign function global variables */
int64_T **h; /* Will point to array of int64_T * pointing to each hashing function */
double **s; /* Will point to array of double * pointing to each sign function */
size_t no_hash_func; /* Will store number of hashing functions */
size_t *len; /* Will point to array of length of each hashing/sign function */

/* Declare output matrix global variables */
double *output_matrix;
int64_T sketch_dim; /* Number of rows in sketched matrix */

/* Declare other global variables */
mwIndex idx; /* Used to access elements on rows of input matrix */
int64_T outer_dim_start; /* See Note 4 */
int64_T outer_dim_end;
int partial_flag; /* Will be set to 0 if entire tensor is sketch, 1 if tensor is sketched piece-by-piece */

/* Declare and define functions */
void compute_row(int64_T prev_sum, double prev_prod, size_t dim) {
	int64_T sum;
	double prod;
	mwIndex i, loop_start, loop_end;
	
	/* This if statement handles cases when we want to sketch a large tensor piece-by-piece */
	if (dim == no_hash_func - 1 && partial_flag == 1) {
		loop_start = outer_dim_start;
		loop_end = outer_dim_end;
	} else {
		loop_start = 0;
		loop_end = len[dim] - 1;
	}
	
	for(i = loop_start; i <= loop_end; ++i) {
		sum = h[dim][i] + prev_sum;
		prod = s[dim][i]*prev_prod;
		if(dim > 0) {
			compute_row(sum, prod, dim - 1);
		} else {
			mwIndex target_row;
			mwIndex r;
			target_row = (sum - no_hash_func) % sketch_dim;
			for(r = 0; r < no_rows; ++r) {
				output_matrix[r + target_row*no_rows] += prod*input_matrix[r + idx*no_rows];
			}
			++idx;
		}
	}
}

/* mex interface */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Declare other variables */
	mwIndex i;
	
	/* Get input matrix with its dimensions */
	input_matrix = mxGetPr(prhs[0]);
	dim_array = mxGetDimensions(prhs[0]);
	no_rows = dim_array[0];
	no_cols = dim_array[1];
	
	/* Get hashing/sign functions with their dimensions */
	no_hash_func = mxGetNumberOfElements(prhs[1]);
	h = malloc(no_hash_func*sizeof(int64_T *));
	s = malloc(no_hash_func*sizeof(double *));
	len = malloc(no_hash_func*sizeof(size_t));
	for(i = 0; i < no_hash_func; ++i) {
		h[i] = (int64_T *) mxGetData(mxGetCell(prhs[1], i));
		s[i] = mxGetPr(mxGetCell(prhs[2], i));
		len[i] = mxGetM(mxGetCell(prhs[1], i));
	}
	
	/* Create the output matrix */
	sketch_dim = *((int64_T *) mxGetData(prhs[3]));
	plhs[0] = mxCreateDoubleMatrix(no_rows, sketch_dim, mxREAL);
	output_matrix = mxGetPr(plhs[0]);
	
	/* Load optional parameters if given */
	if (nrhs >= 6) {
		outer_dim_start = *((int64_T *) mxGetData(prhs[4])) - 1;
		outer_dim_end = *((int64_T *) mxGetData(prhs[5])) - 1;
		partial_flag = 1;
	} else {
		partial_flag = 0;
	}
	
	/* Compute output matrix */
	idx = 0;
	compute_row(0, 1.0, no_hash_func - 1);
	
	/* Free dynamically allocated memory */
	free(len);
	free(s);
	free(h);
}

