function loss = PSO(x1,x2,test_x1,test_x2, fig)
    mu1 = mean(x1)';
    mu2 = mean(x2)';
    p1 = datasample(x1,10);
    p2 = datasample(x2,10);
    f = @(a,x) abs(a - x);
    
    min_f = min(x1);
    max_f = max(x1);
    xbg1 = sdpvar(2,1);
    obj = norm(mu2 - xbg1);
    constraints = [xbg1'*cov(x1)*xbg1 <= 1];%, min_f(1) <= xbg1(1) <= max_f(1), min_f(2) <= xbg1(2) <= max_f(2)]; 
    options=sdpsettings('verbose',0,'solver','scip');
    optimize(constraints,obj,options)
    fprintf('x value: %.5f\n', value(xbg1));

    min_f = min(x2);
    max_f = max(x2);
    xbg2 = sdpvar(2,1);
    obj = norm(mu1 - xbg2);
    constraints = [xbg2'*cov(x2)*xbg2 <= 1];%, min_f(1) <= xbg2(1) <= max_f(1), min_f(2) <= xbg2(2) <= max_f(2)]; 
    options=sdpsettings('verbose',0,'solver','scip');
    optimize(constraints,obj,options)
    fprintf('x value: %.5f\n', value(xbg2));


    xbg1 = [value(xbg1(1)) value(xbg1(2))];
    xbg2 = [value(xbg2(1)) value(xbg2(2))];
%     xbg1 = QCQP_PSO(f,cov(x1),300,p1',mu2);
% 
%     xbg2 = QCQP_PSO(f,cov(x2),300,p2',mu1);
    % 
    if xbg2(2)-xbg1(2) == 0
        hyper_plane_grad = 1;
    else
        hyper_plane_grad = -1*(xbg2(1)-xbg1(1))/(xbg2(2)-xbg1(2));
    end
    mid_point = [(xbg2(1)+xbg1(1))/2 (xbg2(2)+xbg1(2))/2]; 
    const = mid_point(2) - hyper_plane_grad * mid_point(1);

    hyper_plane_x = [min(min(x1(:,1)),min(x2(:,1))):max(max(x1(:,1)),max(x2(:,1)))];
    hyper_plane = hyper_plane_grad * hyper_plane_x + const;
%     figure(fig)
%     plot(x1(:,1),x1(:,2),'o')
%     hold on
%     plot(x2(:,1),x2(:,2),'+')
%     plot(hyper_plane_x,hyper_plane)

    [test_x1,test_x2] = scale_data(test_x1,test_x2);
    
    loss = 0;
%     x1(1,1)*hyper_plane_grad + const
%     x1(1,1)
    if (mu1(1,1)*hyper_plane_grad + const) < mu1(2,1)
%         disp('x1 above')
        s = size(test_x1);
        for i=1:s(1)
            if (test_x1(i,1)*hyper_plane_grad + const) > test_x1(i,2)
                loss = loss + 1;
            end
        end
    else
%         disp('x1 below')
        s = size(test_x1);
        for i=1:s(1)
            if (test_x1(i,1)*hyper_plane_grad + const) < test_x1(i,2)
                loss = loss + 1;
            end
        end
    end
%     (x2(1,1)*hyper_plane_grad + const) 
%     x2(1,1)
    if (mu2(1,1)*hyper_plane_grad + const) < mu2(2,1)
%         disp('x2 above')
        s = size(test_x2);
        for i=1:s(1)
            if (test_x2(i,1)*hyper_plane_grad + const) > test_x2(i,2)
                loss = loss + 1;
            end
        end
    else
%         disp('x2 below')
        s = size(test_x2);
        for i=1:s(1)
            if (test_x2(i,1)*hyper_plane_grad + const) < test_x2(i,2)
                loss = loss + 1;
            end
        end
    end
%     figure(fig+10)
%     plot(test_x1(:,1),test_x1(:,2),'o')
%     hold on
%     plot(test_x2(:,1),test_x2(:,2),'+')
%     plot(hyper_plane_x,hyper_plane)
end