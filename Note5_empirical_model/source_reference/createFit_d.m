function [fitresult, gof] = createFit_d(d_x_to_fit, d_fo_fit_smooth)
%CREATEFIT(D_X_TO_FIT,D_FO_FIT_SMOOTH)
%  创建一个拟合。
%
%  要进行 '无标题拟合 1' 拟合的数据:
%      X 输入: d_x_to_fit
%      Y 输出: d_fo_fit_smooth
%  输出:
%      fitresult: 表示拟合的拟合对象。
%      gof: 带有拟合优度信息的结构体。
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 08-Feb-2025 22:16:40 自动生成


%% 拟合: '无标题拟合 1'。
[xData, yData] = prepareCurveData( d_x_to_fit, d_fo_fit_smooth );

% 设置 fittype 和选项。
ft = fittype( 'a*exp(-x/b)', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [0.460443841792967 0.946148391880459];

% 对数据进行模型拟合。
[fitresult, gof] = fit( xData, yData, ft, opts );

% 绘制数据拟合图。
figure( 'Name', '无标题拟合 1' );
h = plot( fitresult, xData, yData );
legend( h, 'd_fo_fit_smooth vs. d_x_to_fit', '无标题拟合 1', 'Location', 'NorthEast', 'Interpreter', 'none' );
% 为坐标区加标签
xlabel( 'd_x_to_fit', 'Interpreter', 'none' );
ylabel( 'd_fo_fit_smooth', 'Interpreter', 'none' );
grid on


