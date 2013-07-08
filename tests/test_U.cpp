#include "../spin.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <ctime>
int main() {
	mt19937 engine(static_cast<unsigned long>(time(nullptr)));
	poisson_distribution<int> dim_dist(10);
	uniform_real_distribution<double> t_dist(0,1);
	while(true) {
		int dim = dim_dist(engine);
		if(!dim)
			continue;
		double t = t_dist(engine);
		cout << "dimension: " << dim << endl;
		MatrixXcd mat = MatrixXcd::Random(dim,dim);
		MatrixXcd mat1 = mat.adjoint();
		mat += mat1;
		/* calculate by Operator::U */
		Operator op(0,mat);
		auto U = op.U()(t);
		/* calculate by Eigen's matrix exponential */
		MatrixXcd mat_exp = (-1_i*mat*t/hbar).exp();
		/* calculate error */
		MatrixXcd diff = U.matrix()-mat_exp;
		double diff_sum = 0;
		for(int i=0;i<dim;i++)
			for(int j=0;j<dim;j++)
				diff_sum += abs(diff(i,j));
		cout << "error is:" << diff_sum << endl << endl;
	}
}