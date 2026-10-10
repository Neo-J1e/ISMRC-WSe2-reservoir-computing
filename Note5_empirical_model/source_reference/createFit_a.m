function [fitresult, gof] = createFit_a(I_laser, a_all)
%CREATEFIT(I_LASER,A_ALL)
%  创建一个拟合。
%
%  要进行 'a_fitting' 拟合的数据:
%      X 输入: I_laser
%      Y 输出: a_all
%  输出:
%      fitresult: 表示拟合的拟合对象。
%      gof: 带有拟合优度信息的结构体。
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 08-Feb-2025 21:59:42 自动生成


%% 拟合: 'a_fitting'。
[xData, yData] = prepareCurveData( I_laser, a_all );

% 设置 fittype 和选项。
ft = fittype( 'exp2' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.60749697772111 0.000687895729753431 -22.4114746785781 -0.101055493783413];

% 对数据进行模型拟合。
[fitresult, gof] = fit( xData, yData, ft, opts );

% 绘制数据拟合图。
figure( 'Name', 'a_fitting' );
h = plot( fitresult, xData, yData );
legend( h, 'a_all vs. I_laser', 'a_fitting', 'Location', 'NorthEast', 'Interpreter', 'none' );
% 为坐标区加标签
xlabel( 'I_laser', 'Interpreter', 'none' );
ylabel( 'a_all', 'Interpreter', 'none' );
grid on


