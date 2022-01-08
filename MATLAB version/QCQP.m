clc;
clear all;
a = [1; 1];
A = eye(2);
errors = [];
for i=1:500
    mu1 = [-0.25;0];
    sigma1 = [0.1 0.1;0.1 0.1];
    x1 = mvnrnd(mu1,sigma1,100);
    mu2 = [0;-0.25];
    % sigma2 = [0.1 0.1;0.1 0.1];
    % x2 = mvnrnd(mu2,sigma2,100);

    f = @(a,x) a'*x;
    xbg1 = QCQP_PSO(f,A,300,x1',mu2);
    errors(i) = (abs(0.7071 - xbg1(1)) + abs(0.7071 - xbg1(2)))/2;
end
min(errors)