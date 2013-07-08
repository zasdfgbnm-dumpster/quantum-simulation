#include "../spin.hpp"
#include <unsupported/Eigen/KroneckerProduct>
#include <iostream>
#include <random>
#include <ctime>

/* the test runs like this:
 * 1. Determine the structure of Hilbert space
 * A random number is generated to determine the value of nr_spaces,
 * which represents the number of subspaces the total Hilbert space
 * contains.  The random numbers will be generated to decide the
 * dimension of each subspace, these dimensions will be stored in
 * array dim_spaces.
 * 2. Calculate and compare the result with Eigen library
 * Calculate A1*A2*...*Ai+B1*B2*...*Bj+... using class Operator,
 * then compare the result with the result gotten by Eigen library.
 * A1,A2,...Ai,B1,B2,...Bj,... are all random matrices.
 * The number of terms to be added is generated by random generator
 * and stored in nr_terms.  The number of factor of each term is generated
 * and stored in nr_factors.  The subspace of each factor is determined
 * by random generator.
 */

 /* return a random operator located in a random subspace and write its matrix to mat */
 /* lsize[k] stores the dimension of the product space of spaces numbered 1,2,...,k */
Operator gen(MatrixXcd &mat,int nr_spaces,int dim_spaces[],int lsize[],mt19937 &engine) {
	int total_size = lsize[nr_spaces-1];
	uniform_int_distribution<int> subspace_dist(0,nr_spaces-1);
	int subspace  = subspace_dist(engine);
	int dim = dim_spaces[subspace];
	int ldim = lsize[subspace]/dim;
	int rdim = total_size/lsize[subspace];
	MatrixXcd mat1 = MatrixXcd::Random(dim,dim);
	MatrixXcd mat2;
	kroneckerProduct(MatrixXcd::Identity(ldim,ldim),mat1,mat2);
	kroneckerProduct(mat2,MatrixXcd::Identity(rdim,rdim),mat);
	return Operator(subspace,mat1);
}
 
int main(){

	/* distributions */
	mt19937 engine(static_cast<unsigned long>(time(nullptr)));
	poisson_distribution<int> nr_spaces_dist(5);  /* the distribution of number of subspaces */
	poisson_distribution<int> dim_space_dist(3);  /* the distribution of dimension per subspace */
	poisson_distribution<int> nr_terms_dist(100); /* the distribution of number of terms */
	poisson_distribution<int> nr_factor_dist(2);  /* the distribution of number of factor per term */
	
	/* run test */
	while(true) {
		/* initialize parameters in this test */
		int nr_spaces = 1+nr_spaces_dist(engine);
		int nr_terms = 1+nr_terms_dist(engine);
		int dim_spaces[nr_spaces];
		int nr_factors[nr_terms];
		generate(dim_spaces,dim_spaces+nr_spaces,bind(plus<int>(),2,bind(dim_space_dist,engine)));
		generate(nr_factors,nr_factors+nr_terms,bind(plus<int>(),1,bind(nr_factor_dist,engine)));
		int lsize[nr_spaces]; /* lsize[k] stores the dimension of the product space of spaces numbered 1,2,...,k */
		partial_sum(dim_spaces,dim_spaces+nr_spaces,lsize,multiplies<int>());
		int total_size = lsize[nr_space-1]
		/* output Hilbert space structure */
		cout << "Hilbert space structure:" << endl;
		for(auto i : dim_spaces)
			cout << i << ' ';
		cout << endl << endl;
		/* do calculation */
		Operator result_op;
		MatrixXcd result_mat = MatrixXcd::Zeros(total_size,total_size);
		for(int i=0;i<nr_terms;i++) {
			MatrixXcd term_mat;
			Operator term_op = gen(term_mat,nr_spaces,dim_spaces,lsize,engine);
			for(int j=1;j<nr_factors;j++) {
				MatrixXcd mat1;
				Operator op1 = gen(mat1,nr_spaces,dim_spaces,lsize,engine);
				term_mat *= mat1;
				term_op *= op1;
			}
			result_op += term_op;
			result_mat += term_mat;
		}
	}
}