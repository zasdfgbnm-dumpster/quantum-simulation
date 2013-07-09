#include <Eigen/Sparse>
#include "../spin.hpp"
#include <unsupported/Eigen/KroneckerProduct>
#include <ctime>
#include <iostream>
#include <functional>
int main() {
	mt19937 engine(static_cast<unsigned long>(time(nullptr)));
	poisson_distribution<int> nr_space_dist(3);
	poisson_distribution<int> nr_dim_dist(2);
	while(true){
		/* generate a random structure of Hilbert space */
		int nr_space = nr_space_dist(engine);
		if(!nr_space)
			continue;
		int dim_spaces[nr_space];
		generate(dim_spaces,dim_spaces+nr_space,[&engine,&nr_dim_dist]
		{
			while(true){
				int dim=nr_dim_dist(engine);
				if(dim) return dim;
			}
		});
		cout << "Structure of Hilbert space:" << endl;
		for(auto i:dim_spaces)
			cout << i << " ";
		cout << endl;
		/* generate random matrices and calculate their corresponding operator */
		MatrixXcd mat[nr_space];
		transform(dim_spaces,dim_spaces+nr_space,mat,[](int dim){ return MatrixXcd::Random(dim,dim); });
		Operator op(0,mat[0]);
		for(int i=1;i<nr_space;i++)
			op *= Operator(i,mat[i]);
		/* determine which subspace will be traced */
		uniform_int_distribution<int> subspace_dist(0,nr_space-1);
		int subspace = subspace_dist(engine);
		cout << "subspace to be traced: " << subspace << endl;
		/* calculate partial trace by Eigen library */
		MatrixXcd result_eigen = MatrixXcd::Identity(1,1);
		MatrixXcd temp2;
		for(int i=0;i<nr_space;i++) {
			if(i==subspace)
				temp2 = result_eigen*(mat[i].trace());
			else
				kroneckerProduct(result_eigen,mat[i],temp2);
			result_eigen = temp2;
		}
		/* calculate difference between Eigen's result and Operator::tr()'s result */
		MatrixXcd diff = result_eigen - tr(op,subspace).matrix();
		double diff_sum = 0;
		int dim = diff.cols();
		for(int i=0;i<dim;i++)
			for(int j=0;j<dim;j++)
				diff_sum += abs(diff(i,j));
		cout << "error is:" << diff_sum << endl << endl;
	}
}