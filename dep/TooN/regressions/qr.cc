#include "regressions/regression.h"
using namespace TooN;
using namespace std;

int main()
{
	Matrix<3,4> m;
	
	m = Data(5.4388593399963903e-01,
9.9370462412085203e-01,
1.0969746452319418e-01,
4.4837291206649532e-01,
7.2104662057981139e-01,
2.1867663239963386e-01,
6.3591370975105699e-02,
3.6581617683817125e-01,
5.2249530577710213e-01,
1.0579827325022817e-01,
4.0457999585762583e-01,
7.6350464084881342e-01);
	
	cout << setprecision(20) << scientific;
	cout << m << endl;

	QR_Lapack<3, 4> q(m);

	cout << q.get_R() << endl;
	cout << q.get_Q() << endl;

	cout << q.get_Q() * q.get_R() - m << endl;

}

