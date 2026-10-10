function NR = search_para(fac_range,jj_range,step,ML,data_all,rowmin,rowmax,Label,if_wireless)
if nargin == 8
    if_wireless = 0;
end
for fac = fac_range
    for jj = jj_range

        curve_v = (rescale(data_all,"InputMin",rowmin,"InputMax",rowmax)-0.5)*fac*2;
        tanh_v = 1+tanh(curve_v+jj);     %true

        states_train = [];
        for i = 1:step
            a = tanh_v(:, ML*(i-1)+1:ML*i);
            states_train(:,i) = a(:);
        end
        % disp(size(states_train))
        states_test =[];
        for i = step+1:2*step
            a = tanh_v(:, ML*(i-1)+1:ML*i);
            states_test(:,i-step) = a(:);
        end
        % disp(size(states_test))
        states = [ones(1,2*step);states_train,states_test];

        Wout = Label*pinv(states);
        Out = Wout*states;
if if_wireless == 0
        NRMSE_all_train = sqrt(mean((Out(1:end)-Label(1:end)).^2)./var(Label(1:end)));
else

    RES = Out == max(Out);
    comp = sum(Label == RES);
    counter_res = histcounts(comp);
    NRMSE_all_train = counter_res(1)/sum(counter_res);

end

        NR(fac_range == fac,jj_range == jj) = NRMSE_all_train;

    end
end

end