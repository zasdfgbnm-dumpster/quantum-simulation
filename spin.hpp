#ifndef SPIN_HPP
#define SPIN_HPP

#include <complex>
#include <cmath>
#include <numeric>
#include <Eigen/Eigen>
using namespace std;
using namespace Eigen;

//-----------------------------------------------------------------------------------

/* mathematic constants */
/* pi */
#ifndef M_PI
#define M_PI 2*cos(-1)
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
	
	/* The variable named "subspace_dim" store the dimision of subspaces which this operator is in.
	 * the subspaces is numbered one by one from zero.  The value of subspace_dim[a] is the dimision
	 * of subspace numbered a.  This means that, for an operator in subspace numbered 6 and 7, subspace_dim
	 * will have 8 elements, the first 6 of which have no use.  In this case, the values of these 6
	 * elements must be set to any integer less than or equal to 0.  The reason for designing like that
	 * is for simplicity, because we won't have a large amount subspaces because of the dificulty in quantum
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
	 * the dimision of B is given by the parameter "dimision"
	 */
	Operator expand(int subspace,int dimision) const {}
	
public:
	
	Operator(vector<int> subspace_dim,MatrixXcd matrix):subspace_dim(subspace_dim),mat(matrix){}
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
		/* here we define two terms: lspace, rspace and tspace
		 * lspace is a Hilbert space which is the direct product of space 0,1,2,...,(subspace-1)
		 * rspace is a Hilbert space which is the direct product of space (subspace+1),(subspace+2),...,n
		 * tspace is the Hilbert space to be traced
		 */
		int dim;		/* dimision of tspace */
		int new_dim;	/* dimision of the result (i.e. the direct product space of lspace and rspace) */
		int ldim;		/* dimision of lspace */
		int rdim;		/* dimision of rspace */
		int rdim2;		/* dimision of the direct product space of tspace and rspace */
		/* if no information stored, return *this */
		if(subspace>=subspace_dim.size())
			return *this;
		dim = subspace_dim[subspace];
		if(dim<=0)
			return *this;
		/* calculate new_dim,ldim, rdim and rdim2 */
		new_dim = mat.cols()/dim;
		ldim = accumulate(subspace_dim.begin(),subspace.begin()+subspace,1,
						  [](int a,int b){ return (a<=0?1:a)*(b<=0?1:b); });
		rdim = new_dim/ldim;
		rdim2 = rdim*dim;
		/* calculate the result matrix */
		MatrixXcd ret(new_dim,new_dim);
		for(int i=0;i<new_dim;i++) {
			for(int j=0;j<new_dim;j++) {
				/* (i1,j1) is (i,j)'s coordinate in lspace */
				int i1 = i/rdim;
				int j1 = j/rdim;
				/* (i2,j2) is (i,j)'s coordinate in rspace */
				int i2 = i%rdim;
				int j2 = j%rdim;
				ret(i,j) = 0;
				for(int k=0;k<ldim;k++) {/* (k,k) is the coordinate in tspace */
					int i3 = i1*rdim2+k*rdim+i2;
					int j3 = j1*rdim2+k*rdim+j2
					ret(i,j) += mat(i3,j3);
				}
			}
		}
		return ret;
	}
	
	/* arithmetic of operator */
	Operator operator+(const Operator &rhs) const {
		bool lhs_needs_expand;
	}
	Operator operator-(const Operator &rhs) const {
	}
	Operator operator*(const Operator &rhs) const {
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
};

/* instead of writing op1.tr(...) we can also write tr(op1,...) */
template <typename ... Tn>
Operator tr(const Operator &op,Tn ... args) {
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

#endif
