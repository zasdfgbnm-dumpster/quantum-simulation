#ifndef SPIN_HPP
#define SPIN_HPP

#include <complex>
#include <cmath>
#include <numeric>
#include <functional>
#include <algorithm>
#include <tuple>
#include <Eigen/Eigen>
using namespace std;
using namespace Eigen;

//-----------------------------------------------------------------------------------

/* mathematic constants */
/* pi */
#ifndef M_PI
#define M_PI acos(-1)
#endif
constexpr double pi = M_PI;

/* physics constants */
constexpr double hbar = 1; /* for simply we set hbar=1 here */

//-----------------------------------------------------------------------------------

/* useful literal constants ( a C++11-only feature ) */

/* automatically convert these Units to SI */
double operator "" _Hz (long double f) {
	return static_cast<double>(f);
}
double operator "" _Hz (unsigned long long f) {
	return static_cast<double>(f);
}
double operator "" _KHz (long double f) {
	return 1e3*static_cast<double>(f);
}
double operator "" _KHz (unsigned long long f) {
	return 1e3*static_cast<double>(f);
}
double operator "" _kHz (long double f) {
	return 1e3*static_cast<double>(f);
}
double operator "" _kHz (unsigned long long f) {
	return 1e3*static_cast<double>(f);
}
double operator "" _MHz (long double f) {
	return 1e6*static_cast<double>(f);
}
double operator "" _MHz (unsigned long long f) {
	return 1e6*static_cast<double>(f);
}
double operator "" _GHz (long double f) {
	return 1e9*static_cast<double>(f);
}
double operator "" _GHz (unsigned long long f) {
	return 1e9*static_cast<double>(f);
}
double operator "" _THz (long double f) {
	return 1e12*static_cast<double>(f);
}
double operator "" _THz (unsigned long long f) {
	return 1e12*static_cast<double>(f);
}
double operator "" _ns (long double f) {
	return static_cast<double>(f)/1e9;
}
double operator "" _ns (unsigned long long f) {
	return static_cast<double>(f)/1e9;
}
double operator "" _us (long double f) {
	return static_cast<double>(f)/1e6;
}
double operator "" _us (unsigned long long f) {
	return static_cast<double>(f)/1e6;
}
double operator "" _ms (long double f) {
	return static_cast<double>(f)/1e3;
}
double operator "" _ms (unsigned long long f) {
	return static_cast<double>(f)/1e3;
}
double operator "" _T (long double f) {
	return static_cast<double>(f);
}
double operator "" _T (unsigned long long f) {
	return static_cast<double>(f);
}
double operator "" _G (long double f) {
	return static_cast<double>(f)/1e4;
}
double operator "" _G (unsigned long long f) {
	return static_cast<double>(f)/1e4;
}

/* an easier way to input complex number */ 
std::complex<double> operator "" _i (long double f) {
	return std::complex<double>(0,static_cast<double>(f));
}
std::complex<double> operator "" _i (unsigned long long f) {
	return std::complex<double>(0,static_cast<double>(f));
}

//-----------------------------------------------------------------------------------

/* useful matrix functions */

/* calculate exp(a*H) where is a Hermitian matrix */
template <typename matrix>
matrix exp_aH(typename matrix::Scalar a,const matrix &H){
	int n = H.rows();
	SelfAdjointEigenSolver<matrix> es(H);
	matrix diag = matrix::Zero(n,n);
	for(int i=0;i<n;i++)
		diag(i,i) = exp(a*es.eigenvalues()[i]);
	matrix V = es.eigenvectors();
	return V*diag*V.adjoint();
}

//-----------------------------------------------------------------------------------

/* C++ capsulation of operators in physics */

/* Operator class */
class Operator {

	bool null_identity = false;
	/* The variable named "subspace_dim" store the dimension of subspaces which this operator is in.
	 * the subspaces is numbered one by one from zero.  The value of subspace_dim[a] is the dimension
	 * of subspace numbered a.  This means that, for an operator in subspace numbered 6 and 7, subspace_dim
	 * will have 8 elements, the first 6 of which have no use.  In this case, the values of these 6
	 * elements must be set to any integer less than or equal to 0.  The reason for designing like that
	 * is for simplicity, because we won't have a large amount subspaces because of the difficulty in quantum
	 * many-body problem.  So the subspaces must be numbered one by one from zero, giving a subspace a large
	 * number won't lead to mistakes in the result, but will cause serious waste in memory and computing time.
	 */
	vector<int> subspace_dim;
	
	/* the variable "mat" stores the corresponding matrix of this operator.  Subspaces will be ordered by its number
	 * for example the operator B*A where B is in space 1 and A is in space 0, the matrix of B*A will be A@B
	 * where @ stands for kronecker product
	 */
	MatrixXcd mat; 
	
	/* expand current operator to a larger Hilbert space
	 * the result operator will be in the direct product space of A and B
	 * where A is current operator's space and B is the space specified by parameter "subspace"
	 * the dimension of B is given by the parameter "dimension"
	 */
	Operator expand(int subspace,int dimension) const {
		vector<int> dim_info = subspace_dim;
		if(subspace+1>static_cast<signed int>(dim_info.size()))
			dim_info.resize(subspace+1,0);
		if(dim_info[subspace]>0)
			throw "Operator::expand(): already in subspace";
		dim_info[subspace] = dimension;
		/* here we define several terms: non-empty, lspace, rspace and espace
		 * we say a subspace numbered n is non-empty if subspace_dim[n]>0 
		 * lspace is the direct product space of non empty spaces numbered 0,1,2,...,(subspace-1)
		 * rspace is the direct product space of non empty spaces numbered (subspace+1),(subspace+2),...,n
		 * espace is the space numbered "subspace"(the parameter given)
		 */
		int new_dim;	/* dimension of the result (i.e. the direct product space of lspace, espace and rspace) */
		int ldim;		/* dimension of lspace */
		int rdim;		/* dimension of rspace */
		int rdim2;		/* dimension of the direct product space of espace and rspace */
		/* calculate new_dim, ldim, rdim and rdim2 */
		int mcol = mat.cols();
		/* if the operator before expand is null */
		if(mcol==0)
			return null_identity?Operator(subspace,MatrixXcd::Identity(dimension,dimension)):Operator(subspace,MatrixXcd::Zero(dimension,dimension));
		/* if the operator before expand is not null */
		new_dim = mcol*dimension;
		ldim = accumulate(dim_info.begin(),dim_info.begin()+subspace,1,
						  [](int a,int b){ return (a<=0?1:a)*(b<=0?1:b); });
		rdim2 = new_dim/ldim;
		rdim = rdim2/dimension;
		MatrixXcd ret(new_dim,new_dim);
		/* calculate new elements */
		#pragma omp parallel for
		for(int i=0;i<new_dim;i++) {
			for(int j=0;j<new_dim;j++) {
				/* (i1,j1) is (i,j)'s coordinate in lspace */
				int i1 = i/rdim2;
				int j1 = j/rdim2;
				/* (i2,j2) is (i,j)'s coordinate in espace */
				int i2 = i%rdim2/rdim;
				int j2 = j%rdim2/rdim;
				/* (i3,j3) is (i,j)'s coordinate in rspace */
				int i3 = i%rdim2%rdim;
				int j3 = j%rdim2%rdim;
				/* (i4,j4) is (i,j)'s corresponding coordinate in the direct procuct space of lspace and rspace */
				int i4 = i1*rdim+i3;
				int j4 = j1*rdim+j3;
				ret(i,j) = (i2!=j2?0:mat(i4,j4));
			}
		}
		return Operator(dim_info,ret);
	}
	
	/* expand current operator to the product space of op(given by parameter) and this operator
	 * note that the product may not be direct product
	 */ 
	Operator expand(const Operator &op) const {
		Operator ret = *this;
		vector<int> target_dim = subspace_dim;
		int op_sz = op.subspace_dim.size();
		if(static_cast<signed int>(target_dim.size())<op_sz)
			target_dim.resize(op_sz,0);
		auto it1 = target_dim.begin();
		auto it2 = op.subspace_dim.begin();
		while(it2!=op.subspace_dim.end()){
			if(*it1<=0&&*it2<=0)
				goto end;
			if(*it1==*it2)
				goto end;
			if(*it1>0&&*it2>0)
				throw "Operator::expand(): dimension information mismatch";
			if(*it2>0)
				ret = ret.expand(it1-target_dim.begin(),*it2);
		end:
			++it1;
			++it2;
		}
		return ret;
	}
	
public:
	
	Operator() = default;
	Operator(int i) {
		if(i==1) null_identity=true;
		else if(i!=0) throw "Operator::Operator(): null operator must be zero or identity";
	}
	Operator(vector<int> subspace_dim,MatrixXcd matrix):subspace_dim(subspace_dim),mat(matrix){
		if(subspace_dim.size()==0)
			throw "Operator::Operator(): the operator must be in at least one subspace";
		int dim = accumulate(subspace_dim.begin(),subspace_dim.end(),1,
							 [](int a,int b){ return (a<=0?1:a)*(b<=0?1:b); });
		if(dim!=matrix.cols()||dim!=matrix.rows())
			throw "Operator::Operator(): matrix size and dimension information mismatch";
	}
	/* initialize an operator in a single subspace */
	Operator(int subspace,const MatrixXcd &matrix):subspace_dim(subspace+1,0),mat(matrix) {
		if(subspace<0)
			throw "Operator::Operator(): subspace can't be negative";
		int dim1,dim2;
		dim1 = mat.rows();
		dim2 = mat.cols();
		if(dim1!=dim2)
			throw "Operator::Operator(): matrix is not square ";
		subspace_dim[subspace] = dim1;
	}
	
	/* return the matrix of this operator*/
	const MatrixXcd &matrix() const { return mat; }
	
	/* trace of the operator */
	complex<double> tr() const {
		return mat.trace();
	}
	
	/* partial trace of the operator 
	 * this function make use of C++11's feature of variadic templates.
	 * this feature makes it possible to pass arbitary number of parameters to function
	 * 
	 * to use this function, just write:
	 * operator1.tr(subspace1,subspace2,subspace3,....)
	 * 
	 * to get more information about variadic templates,
	 * see Gregoire, Solter and Kleper's book :
	 * Professional C++, Second Edition  chapter 20.6
	 */
	template <typename ... Tn>
	Operator tr(int subspace,Tn ... args) const {
		return tr(subspace).tr(args...);
	}
	Operator tr(int subspace) const {
		/* here we define several terms: non-empty, lspace, rspace and tspace
		 * we say a subspace numbered n is non-empty if subspace_dim[n]>0 
		 * lspace is the direct product space of non empty spaces numbered 0,1,2,...,(subspace-1)
		 * rspace is the direct product space of non empty spaces numbered (subspace+1),(subspace+2),...,n
		 * tspace is the Hilbert space to be traced
		 */
		int dim;		/* dimension of tspace */
		int new_dim;	/* dimension of the result (i.e. the direct product space of lspace and rspace) */
		int ldim;		/* dimension of lspace */
		int rdim;		/* dimension of rspace */
		int rdim2;		/* dimension of the direct product space of tspace and rspace */
		/* if no information stored, return *this */
		if(subspace>=static_cast<signed int>(subspace_dim.size()))
			return *this;
		dim = subspace_dim[subspace];
		if(dim<=0)
			return *this;
		/* calculate new_dim, ldim, rdim and rdim2 */
		new_dim = mat.cols()/dim;
		ldim = accumulate(subspace_dim.begin(),subspace_dim.begin()+subspace,1,
						  [](int a,int b){ return (a<=0?1:a)*(b<=0?1:b); });
		rdim = new_dim/ldim;
		rdim2 = rdim*dim;
		/* calculate the result matrix */
		MatrixXcd ret(new_dim,new_dim);
		#pragma omp parallel for
		for(int i=0;i<new_dim;i++) {
			for(int j=0;j<new_dim;j++) {
				/* (i1,j1) is (i,j)'s coordinate in lspace */
				int i1 = i/rdim;
				int j1 = j/rdim;
				/* (i2,j2) is (i,j)'s coordinate in rspace */
				int i2 = i%rdim;
				int j2 = j%rdim;
				ret(i,j) = 0;
				for(int k=0;k<dim;k++) {/* (k,k) is the coordinate in tspace */
					int i3 = i1*rdim2+k*rdim+i2;
					int j3 = j1*rdim2+k*rdim+j2;
					ret(i,j) += mat(i3,j3);
				}
			}
		}
		/* generate the new operator */
		vector<int> dim_info = subspace_dim;
		dim_info[subspace] = 0;
		return Operator(dim_info,ret);
	}
	
	/* arithmetic of operator */
	Operator operator+(const Operator &rhs) const {
		Operator _lhs = expand(rhs);
		Operator _rhs = rhs.expand(*this);
		return Operator(_lhs.subspace_dim,_lhs.mat+_rhs.mat);
	}
	Operator operator-(const Operator &rhs) const {
		Operator _lhs = expand(rhs);
		Operator _rhs = rhs.expand(*this);
		return Operator(_lhs.subspace_dim,_lhs.mat-_rhs.mat);
	}
	Operator operator*(const Operator &rhs) const {
		Operator _lhs = expand(rhs);
		Operator _rhs = rhs.expand(*this);
		return Operator(_lhs.subspace_dim,_lhs.mat*_rhs.mat);
	}
	Operator operator*(const complex<double> &c) const {
		return Operator(subspace_dim,c*mat);
	}
	Operator operator/(const complex<double> &c) const {
		return Operator(subspace_dim,mat/c);
	}
	template<typename T>
	Operator &operator+=(const T &rhs) {
		return (*this = operator+(rhs));
	}
	/* a*=b is equivalent to a=a*b which may not equal to a=b*a */
	template<typename T>
	Operator &operator*=(const T &rhs) {
		return (*this = operator*(rhs));
	}
	template<typename T>
	Operator &operator-=(const T &rhs) {
		return (*this = operator-(rhs));
	}
	template<typename T>
	Operator &operator/=(const T &rhs) {
		return (*this = operator/(rhs));
	}
	Operator operator+() const {
		return *this;
	}
	Operator operator-() const {
		return Operator(subspace_dim,-mat);
	}
	
	/* Hermitian conjugate of this operator */
	Operator operator*() const {
		return Operator(subspace_dim,mat.adjoint());
	}
	
	/* calling H.U() returns a function (double->Operator): t->exp(-i*H*t/hbar)
	 * to call U(), H must be a Hermitian operator.  If this condition is violated
	 * you won't get the correct result.
	 * both this function and the function returned has a space complexity N^2
	 */
	function<Operator(double)> U() {
		SelfAdjointEigenSolver<MatrixXcd> es(mat);
		auto eigenvalues  = es.eigenvalues();
		auto eigenvectors = es.eigenvectors();
		auto dim_info = subspace_dim;
		return [eigenvalues,eigenvectors,dim_info](double t) -> Operator {
			int n = eigenvalues.size();
			MatrixXcd ret = MatrixXcd::Zero(n,n);
			for(int i=0;i<n;i++)
				ret(i,i) = exp(-1_i*t*eigenvalues[i]/hbar);
			ret = eigenvectors*ret*eigenvectors.adjoint();
			return Operator(dim_info,ret);
		};
	}

	/* Solve Liouville Equation i*hbar*(partial rho/partial t) = [H,rho].
	 * The method to solve this equation is forward difference quotient.
	 * To solve Liouville Equation, the first step is to create an object of LiouvilleEqSolver.
	 * The constructor of LiouvilleEqSolver receives three parameters. The first is the function
	 * which emit H from t.  It must be a function taking a double and return an Operator.
	 * The second parameter is the density operator at time 0.  The third parameter is the step.
	 * The fourth parameter is a vector<double> storing time at which the density operator will
	 * be calculated.  The return value is a vector<Operator> stores density operator at time stored
	 * in the parameter time.  The returned density operator is sorted by time increasing.
	 */
	static vector<Operator> SolveLiouvilleEq(function<Operator(double)> Ht,Operator rho0,double step,vector<double> time,function <void(double)> callback=[](double){}) {
		sort(time.begin(),time.end());
		vector<Operator> ret;
		double t_ = 0;
		for(auto i: time) {
			int n = static_cast<int>((i-t_)/step);
			t_ += n*step;
			Operator H;
			while(n--) {
				H = Ht(t_-(n+1)*step);
				rho0 += (H*rho0-rho0*H)*complex<double>(step,0)/(1_i*hbar);
			}
			H = Ht(t_);
			Operator rhot = rho0 + (H*rho0-rho0*H)*complex<double>(i-t_,0)/(1_i*hbar);
			ret.push_back(rhot);
			callback(i);
		}
		return ret;
	}
};
Operator operator*(complex<double> c,const Operator op){
	return op*c;
}

/* instead of writing op1.tr(...) we can also write tr(op1,...) */
template <typename ... Tn>
auto tr(const Operator &op,Tn ... args) -> decltype(op.tr(args...)) {
	return op.tr(args...);
}

/* spin operators */
/* related formula (see Zhu Dongpei's textbook of quantum mechanics):
 * <m|Jx|m'> = (hbar/2)  * { sqrt[(j+m)(j-m+1)]*delta(m,m'+1) + sqrt[(j-m)(j+m+1)]*delta(m,m'-1) }
 * <m|Jy|m'> = (hbar/2i) * { sqrt[(j+m)(j-m+1)]*delta(m,m'+1) - sqrt[(j-m)(j+m+1)]*delta(m,m'-1) }
 * where |m> is engien state of Jz i.e. Jz|m> = m|m>
 */
Operator Sx(int subspace,int dim=2) {
	MatrixXcd mat(dim,dim);
	mat.setZero();
	double j = (dim-1.0)/2;
	for(int i=0;i<dim-1;i++)
		mat(i,i+1) = 0.5*hbar*sqrt((2*j-i)*(i+1));
	for(int i=1;i<dim;i++)
		mat(i,i-1) = 0.5*hbar*sqrt(i*(2*j-i+1));
	return Operator(subspace,mat);
}
Operator Sy(int subspace,int dim=2) {
	MatrixXcd mat(dim,dim);
	mat.setZero();
	double j = (dim-1.0)/2;
	for(int i=0;i<dim-1;i++)
		mat(i,i+1) = -0.5_i*hbar*sqrt((2*j-i)*(i+1));
	for(int i=1;i<dim;i++)
		mat(i,i-1) = 0.5_i*hbar*sqrt(i*(2*j-i+1));
	return Operator(subspace,mat);
}
Operator Sz(int subspace,int dim=2) {
	MatrixXcd mat(dim,dim);
	mat.setZero();
	double j = (dim-1.0)/2;
	for(int i=0;i<dim;i++)
		mat(i,i) = (j-i)*hbar;
	return Operator(subspace,mat);
}

/* zero and identity operator */
Operator O(int subspace,int dim=2) {
	return Operator(subspace,MatrixXcd::Zero(dim,dim));
}
Operator I(int subspace,int dim=2) {
	return Operator(subspace,MatrixXcd::Identity(dim,dim));
}

/* helper function, won't used by user , used by function Op */
void Op_helper(CommaInitializer<MatrixXcd> &initializer,complex<double> arg1) {
	initializer,arg1;
}
template <typename ... Tn>
void Op_helper(CommaInitializer<MatrixXcd> &initializer,complex<double> arg1,Tn ... args) {
	Op_helper((initializer,arg1),args...);
}
/* this function will be used to generate an arbitary dimension operator
 * to generate a N dimension operator in subspace numbered s1 with matrix
 * element e1,e2,e3,...,eN, just write:
 * Op<N>(s1,e1,e2,....,eN);
 */
template <int n,typename ... Tn>
Operator Op(int subspace,complex<double> arg1,Tn ... args) {
	MatrixXcd mat(n,n);
	CommaInitializer<MatrixXcd> initializer = (mat<<arg1);
	Op_helper(initializer,args...);
	return Operator(subspace,mat);
}


#endif
