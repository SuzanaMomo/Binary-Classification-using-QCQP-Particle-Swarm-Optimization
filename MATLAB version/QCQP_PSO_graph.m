function xbg = QCQP_PSO_graph(f,P,T,X,a)
%     f = @(a,x,v) abs(a-x-v);
%     f_val = [];
    V = zeros(size(X));
    xb = zeros(size(X));
    xbg = X(:,1);
    for i=1:length(X)
        xb(:,i) = X(:,i);
%         f(a,xbg)
        if f(a,xbg) > f(a,xb(:,i))
            xbg = xb(:,i);
        end
    end
    w = 0.05;
    c = [0.05 0.05 0.05 0.05];
    t = 0;
    while t < T
        t = t +1;
        r = rand(4,1);
        for i=1:length(X)
%             disp((a - X(:,1)))
%             V(i)
            V(:,i) = w*V(:,i) + c(1)*r(1)*(xb(:,i) - X(:,i)) + c(2)*r(2)*(xbg - X(:,i)) + c(3)*r(3)*(a - X(:,i)) + c(4)*r(4);
        end
        for i=1:length(X)
            xx = X(:,i) + V(:,i);
%             if xx'*P*xx <= 1
                X(:,i) = xx; 
%             end
            if f(a,xb(:,i)) > f(a,X(:,i))
                xb(:,i) = X(:,i);
            end
            if f(a,xbg) > f(a,xb(:,i))
                xbg = xb(:,i);
            end
        end
        if t == 1
            plot(xb(1,:),xb(2,:),'x')
            hold on;
        end
        if t == 2
            plot(xb(1,:),xb(2,:),'+')
            hold on;
        end
    end
end