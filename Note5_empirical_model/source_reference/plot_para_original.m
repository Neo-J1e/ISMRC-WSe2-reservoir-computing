function [fac,jj,NRMSE] = plot_para(NR,fac_range,jj_range,data_all,rowmin,rowmax,ML,step,Label,if_plot,if_wireless)
min(min(NR));
if nargin == 9
    if_plot = 1;
    if_wireless = 0;
end
if nargin == 10
    if_wireless = 0;
end

[row,col] = ind2sub(size(NR),find(NR == min(min(NR)),1));

% figure
% plot(jj_range,NR)
% legend(num2str(fac_range'))
% colororder(slanCM('rainbow',size(fac_range,2)))

fac = fac_range(row);
jj  = jj_range(col);

curve_v = (rescale(data_all,"InputMin",rowmin,"InputMax",rowmax)-0.5)*fac*2;

tanh_v = 1+tanh(curve_v+jj);     %true


states_train = [];
for i = 1:step
    a = tanh_v(:, ML*(i-1)+1:ML*i);
    states_train(:,i) = a(:);
end
states_train;

X = [ones(1,step);states_train(:,1:end)];


states_test =[];
for i = step+1:2*step
    a = tanh_v(:, ML*(i-1)+1:ML*i);
    states_test(:,i-step) = a(:);
end
states_test;


states = [ones(1,2*step);states_train,states_test];
Wout = Label*pinv(states);
Out = Wout*states;
NRMSE_all_train = sqrt(mean((Out(1:end)-Label(1:end)).^2)./var(Label(1:end)));
if if_plot
figure
imagesc(NR)
hold on
plot(col,row,'r*','MarkerSize',5,'LineWidth',2)
text(col,row,'Best Para Point\rightarrow ','FontName', ...
    'Times new roman','FontWeight','bold', ...
    'HorizontalAlignment','right')
text(col,row+1,['(',num2str(fac),',',num2str(jj),')'],'FontName', ...
    'Times new roman','FontWeight','bold', ...
    'HorizontalAlignment','center')
ylabel('fac')
xlabel('shift')
colorbar
figure_setting('Best Para Map',10,2,10)
grid off

if if_wireless == 1
    RES = Out == max(Out);
    comp = sum(Label == RES);
    counter_res = histcounts(comp);
    NRMSE_all_train = counter_res(1)/sum(counter_res);
end

figure
plot(Label(1:2*step),'LineWidth',1)
hold on
plot(Out(1:2*step),'LineWidth',1)
legend('GT','Result')
figure_setting(['Best Para Result:',num2str(NRMSE_all_train)],10,2,10)
grid off
end

if if_wireless == 1
    RES = Out == max(Out);
    comp = sum(Label == RES);
    counter_res = histcounts(comp);
    NRMSE_all_train = counter_res(1)/sum(counter_res);
end

if nargout == 3
    NRMSE = NRMSE_all_train;
end
end