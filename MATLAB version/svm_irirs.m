clc;
clear all;
data = readtable('iris.csv');
x1_count = 0;
x2_count = 0;
x3_count = 0;
for i=1:height(data)
    class = data.class(i);
    if strcmp(class,'Iris-virginica')
        x3_count = x3_count +1;
    end
    if strcmp(class,'Iris-setosa')
        x1_count = x1_count +1;
    end
    if strcmp(class,'Iris-versicolor')
        x2_count = x2_count +1;
    end
end
x1 = zeros(x1_count,4);
x2 = zeros(x2_count,4);
x3 = zeros(x3_count,4);
% all = zeros(150,4);
x1_count = 1;
x2_count = 1;
x3_count = 1;
for i=1:height(data)
    class = data.class(i);
    if strcmp(class,'Iris-virginica')
        x= table2array(data(i,{'sepal_length','sepal_width','petal_length','petal_width'}));
        x3(x3_count,:) = x;
        x3_count = x3_count + 1;
%         all(i,:) = x;
    end
    if strcmp(class,'Iris-setosa')
        x= table2array(data(i,{'sepal_length','sepal_width','petal_length','petal_width'}));
        x1(x1_count,:) = x;
        x1_count = x1_count + 1;
%         all(i,:) = x;
    end
    if strcmp(class,'Iris-versicolor')
        x= table2array(data(i,{'sepal_length','sepal_width','petal_length','petal_width'}));
        x2(x2_count,:) = x;
        x2_count = x2_count + 1;
%         all(i,:) = x;
    end
end
x2 = vertcat(x2,x3);
x1(:,5) = zeros(length(x1),1);
x2(:,5) = ones(length(x2),1);
data = vertcat(x1,x2);
%%
c = cvpartition(data,'KFold',10)