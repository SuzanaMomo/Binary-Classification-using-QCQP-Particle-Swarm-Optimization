clc;
clear all;
% a = [1; 1]
% x = [1 1; 0 0; 0.5 0.5; 1.5 1.5]
A = eye(2);

mu1 = [8;0];
sigma1 = [2 1;1 2];
x1 = mvnrnd(mu1,sigma1,100);
mu2 = [0;8];
sigma2 = [2 1;1 2];
x2 = mvnrnd(mu2,sigma2,100);
p1 = mvnrnd(mu1,sigma1,10);
p2 = mvnrnd(mu2,sigma2,10);
f = @(a,x) abs(a - x);

min_f = min(x1);
max_f = max(x1);
xbg1 = sdpvar(2,1);
obj = abs(mu2 - xbg1);
constraints = [xbg1'*cov(x1)*xbg1 <= 1]; %, min_f(1) <= x(1) <= max_f(1), min_f(2) <= x(2) <= max_f(2)
options=sdpsettings('verbose',1,'solver','scip');%,'scip.maxtime',10000000, 'scip.maxiter',100000000, 'scip.maxnodes',100000000);   
% options.scip
optimize(constraints,obj,options)
fprintf('x value: %.5f\n', value(xbg1));


min_f = min(x2);
max_f = max(x2);
xbg2 = sdpvar(2,1);
obj = abs(mu1 - xbg2);
constraints = [xbg2'*cov(x2)*xbg2 <= 1]; %, min_f(1) <= x(1) <= max_f(1), min_f(2) <= x(2) <= max_f(2)
options=sdpsettings('verbose',1,'solver','scip');%,'scip.maxtime',10000000, 'scip.maxiter',100000000, 'scip.maxnodes',100000000);   
% options.scip
optimize(constraints,obj,options)
fprintf('x value: %.5f\n', value(xbg2));


xbg1 = [value(xbg1(1)) value(xbg1(2))]
xbg2 = [value(xbg2(1)) value(xbg2(2))]
% xbg1 = QCQP_PSO(f,cov(x1),300,p1',mu2)

% xbg2 = QCQP_PSO(f,cov(x2),300,p2',mu1)

hyper_plane_grad = -1*(xbg2(1)-xbg1(1))/(xbg2(2)-xbg1(2));
mid_point = [(xbg2(1)+xbg1(1))/2 (xbg2(2)+xbg1(2))/2]; 
const = mid_point(2) - hyper_plane_grad * mid_point(1);
hyper_plane_x = [-5:15];
hyper_plane = hyper_plane_grad * hyper_plane_x + const;
plot(x1(:,1),x1(:,2),'o')
hold on
plot(x2(:,1),x2(:,2),'+')
plot(hyper_plane_x,hyper_plane)
% 
