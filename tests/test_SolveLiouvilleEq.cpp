#include <iostream>
#include "spin.hpp"
#include <algorithm>
#include <functional>
#include <ctime>

int main() {
	mt19937 engine(static_cast<unsigned long>(time(nullptr)));
	poisson_distribution<int> dim_dist(2);
	poisson_distribution<int> n_time_dist(10);
	exponential_distribution<double> t_max_dist(1);
	uniform_real_distribution<double> t_dist(0,1);
	while(true) {
		/* generate random H */
		int dim = dim_dist(engine);
		if(!dim)
			continue;
		MatrixXcd mat = MatrixXcd::Random(dim,dim);
		MatrixXcd mat1 = mat.adjoint();
		mat += mat1;
		Operator H(0,mat);
		/* generate random rho0 */
		mat = MatrixXcd::Random(dim,dim);
		mat1 = mat.adjoint();
		mat *= mat1;
		complex<double> tr = mat.trace();
		if(tr==0_i)
			continue;
		mat /= tr;
		Operator rho0(0,mat);
		/* generate random time */
		int n_time = n_time_dist(engine);
		if(n_time==0)
			continue;
		double t_max = t_max_dist(engine);
		vector<double> time(n_time);
		for(auto &i:time) {
			i = t_max*t_dist(engine);
		}
		sort(time.begin(),time.end());
		/* output some information */
		cout << "dimension: " << dim << endl;
		cout << "n_time: " << n_time << endl;
		/* calculate by Operator::U */
		auto U = H.U();
		vector<Operator> U_result(n_time);
		transform(time.begin(),time.end(),U_result.begin(),[rho0,U](double t){ return U(t)*rho0*U(-t); });
		/* calculate by SolveLiouvilleEq */
		vector<Operator> S_result = Operator::SolveLiouvilleEq([H](double){return H;},rho0,t_max/1000000,time);
		/* calculate error */
		double diff_sum = 0;
		auto itU = U_result.begin();
		auto itS = S_result.begin();
		while(itU!=U_result.end()) {
			MatrixXcd diff = (*itU-*itS).matrix();
			for(int i=0;i<diff.size();i++)
				diff_sum += abs(diff(i));
			++itU;
			++itS;
		}
		cout << "error is:" << diff_sum << endl << endl;
	}
}