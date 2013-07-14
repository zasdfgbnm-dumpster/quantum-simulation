#include "../spin.hpp"
#include <iostream>
#include <ctime>
int main() {
	mt19937 engine(static_cast<unsigned long>(time(nullptr)));
	poisson_distribution<int> dim_dist(10);
	poisson_distribution<int> ntests_dist(20);
	uniform_real_distribution<double> t_dist(0,1);
	uniform_int_distribution<int> step_cache_dist(0,50);
	while(true) {
		int dim = dim_dist(engine);
		if(!dim)
			continue;
		cout << "dimension: " << dim << endl;
		MatrixXcd mat = MatrixXcd::Random(dim,dim);
		MatrixXcd mat1 = mat.adjoint();
		mat += mat1;
		Operator op(0,mat);
		int ntests = ntests_dist(engine);
		if(!ntests)
			continue;
		int step_cache = step_cache_dist(engine);
		cout << "number of tests: " << ntests << endl;
		cout << "step_cache: " << step_cache << endl;
		auto U = op.U();
		auto Ut = Operator::Ut([&op](double t){ return op; },1e-3,step_cache);
		for(int k=0;k<ntests;k++) {
			double t = t_dist(engine);
			/* calculate by Operator::U */
			Operator result1 = U(t);
			/* calculate by Operator::Ut */
			Operator result2 = Ut(t);
			/* calculate error */
			MatrixXcd diff = result1.matrix()-result2.matrix();
			double diff_sum = 0;
			for(int i=0;i<dim;i++)
				for(int j=0;j<dim;j++)
					diff_sum += abs(diff(i,j));
			cout << "t= " << t << "\t" << "error is:" << diff_sum << endl;
		}
		cout << endl;
	}
}