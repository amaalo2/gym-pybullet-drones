clear all
close all

states = readNPY('states.npy');
time = readNPY('timestamps.npy');


t = time(1,:);

%ownship 
x_o = reshape(states(1,1,:),1,length(t));
y_o = reshape(states(1,2,:),1,length(t));
z_o = reshape(states(1,3,:),1,length(t));
p_o = [x_o;y_o;z_o];
%intruder
x_i = reshape(states(2,1,:),1,length(t));
y_i = reshape(states(2,2,:),1,length(t));
z_i = reshape(states(2,3,:),1,length(t));
p_i = [x_i;y_i;z_i];

%Create Doi plot

d_oi = zeros(1,length(t));
d2g  = zeros(1,length(t));
goal = [9;0;6];


for i = 1 : length(t)
    d_oi(i) = norm(p_o(:,i)-p_i(:,i));
    d2g(i)  = norm(p_o(:,i)-goal);
end

%Find the local mins
local_mins_doi_idx = islocalmin(d_oi);
local_mins_d2g_idx = islocalmin(d2g);


%Plot the min_doi vs runs
figure
hold on
x_axis = linspace(1,length(t(local_mins_doi_idx)),length(t(local_mins_doi_idx)));

y = ones(1,length(x_axis));
pts = d_oi(local_mins_doi_idx);
three_sigma_upper = (3*std(d_oi(local_mins_doi_idx))+ mean(d_oi(local_mins_doi_idx)))*y;
%three_sigma_lower = (3*std(d_oi(local_mins_doi_idx))- mean(d_oi(local_mins_doi_idx)))*y;

scatter(x_axis(pts>=1),pts(pts>=1),'filled','b')
scatter(x_axis(pts<1),pts(pts<1),'filled','r')
plot(x_axis,mean(d_oi(local_mins_doi_idx))*y, 'LineWidth',2)
%plot(x_axis,three_sigma_lower, 'LineWidth',2,'Color','k')
plot(x_axis,three_sigma_upper, 'LineWidth',2,'Color','k')


ylabel('$d_{oi}$ (m)','fontsize',15,'Interpreter','latex');
xlabel('Test Case','fontsize',15,'Interpreter','latex');
legend('Sucess','Fail','Mean','3$\sigma$ Bound','Interpreter','Latex' )


%Plot the min_distance vs runs


figure
hold on
x_axis = linspace(1,length(t(local_mins_d2g_idx)),length(t(local_mins_d2g_idx)));

y = ones(1,length(x_axis));
pts = d2g(local_mins_d2g_idx);
three_sigma_upper = (3*std(d2g(local_mins_d2g_idx))+ mean(d2g(local_mins_d2g_idx)))*y;
%three_sigma_lower = (3*std(d2g(local_mins_d2g_idx))- mean(d2g(local_mins_d2g_idx)))*y;

scatter(x_axis(pts>=1),pts(pts>=1),'filled','r')
scatter(x_axis(1<pts& pts<5),pts(1<pts& pts<5),'filled','k')
scatter(x_axis(pts<1),pts(pts<1),'filled','b')
%plot(x_axis,mean(d2g(local_mins_d2g_idx))*y, 'LineWidth',2)
%plot(x_axis,three_sigma_lower, 'LineWidth',2,'Color','k')
%plot(x_axis,three_sigma_upper, 'LineWidth',2,'Color','k')

ylabel('$d_{oi}$ (m)','fontsize',20,'Interpreter','latex');
xlabel('Test Case','fontsize',20,'Interpreter','latex');


breakyaxis([2,8])





