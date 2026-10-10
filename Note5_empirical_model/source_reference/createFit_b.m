function [fitresult, gof] = createFit_b(I_laser, b_all)
%CREATEFIT(I_LASER,B_ALL)
%  创建一个拟合。
%
%  要进行 'b_fitting' 拟合的数据:
%      X 输入: I_laser
%      Y 输出: b_all
%  输出:
%      fitresult: 表示拟合的拟合对象。
%      gof: 带有拟合优度信息的结构体。
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 08-Feb-2025 22:00:23 自动生成


%% 拟合: 'b_fitting'。
[xData, yData] = prepareCurveData( I_laser, b_all );

% 设置 fittype 和选项。
ft = fittype( 'exp1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Robust = 'Bisquare';
opts.StartPoint = [12.6685408923856 -0.0260796558943731];

% 对数据进行模型拟合。
[fitresult, gof] = fit( xData, yData, ft, opts );

% 绘制数据拟合图。
figure( 'Name', 'b_fitting' );
h = plot( fitresult, xData, yData );
legend( h, 'b_all vs. I_laser', 'b_fitting', 'Location', 'NorthEast', 'Interpreter', 'none' );
% 为坐标区加标签
xlabel( 'I_laser', 'Interpreter', 'none' );
ylabel( 'b_all', 'Interpreter', 'none' );
grid on


