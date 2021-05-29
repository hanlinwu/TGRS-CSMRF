function [time,x]=optimation(A,B,m,par_cvx)
Nsum = size(A,1);
mapNum = size(A,2);
alpha1 = par_cvx(1);
alpha2 =  par_cvx(2);
beta1 = par_cvx(3);
clear x sm;

cvx_begin quiet
variable x(Nsum,mapNum)
for k = 1:mapNum
    sm(1,k) = quad_form(m'*x(:,k),eye(Nsum));
end
minimize(alpha1*trace(A*x')+alpha2*trace(B*x')+beta1*sum(sm)+sum_square(x(:)))
subject to
ones(1,mapNum)*x' == ones(1,Nsum);
0 <= x(:) <= 1;
cvx_end
time = cvx_cputime;
end