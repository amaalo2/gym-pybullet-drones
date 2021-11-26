clear all
close all


% Font size, line size, and line width, and intruder states
font_size = 15;
line_size = 15;
line_width = 2;


states = readNPY('states.npy');
time = readNPY('timestamps.npy');
controls = readNPY('controls.npy');

log_freq = 48;  % hz

t = time(1,:);

% states ownship 
x_o = reshape(states(1,1,:),1,length(t));
y_o = reshape(states(1,2,:),1,length(t));
z_o = reshape(states(1,3,:),1,length(t));
p_o = [x_o;y_o;z_o];

vx_o = reshape(states(1,4,:),1,length(t));
vy_o = reshape(states(1,5,:),1,length(t));
vz_o = reshape(states(1,6,:),1,length(t));

roll_o   = reshape(states(1,7,:),1,length(t));
pitch_o   = reshape(states(1,8,:),1,length(t));
yaw_o = reshape(states(1,9,:),1,length(t));

rolld_o = [0,log_freq*(roll_o(2:end)-roll_o(1:end-1))];
pitchd_o = [0,log_freq*(pitch_o(2:end)-pitch_o(1:end-1))];
yawd_o = [0,log_freq*(yaw_o(2:end)-yaw_o(1:end-1))];


%ctrl inputs

m1_o = reshape(states(1,13,:),1,length(t));
m2_o = reshape(states(1,14,:),1,length(t));
m3_o = reshape(states(1,15,:),1,length(t));
m4_o = reshape(states(1,16,:),1,length(t));

%velocity targets

vx_target_o = reshape(controls(1,4,:),1,length(t));
vy_target_o = reshape(controls(1,5,:),1,length(t));
vz_target_o = reshape(controls(1,6,:),1,length(t));


%intruder
x_i = reshape(states(2,1,:),1,length(t));
y_i = reshape(states(2,2,:),1,length(t));
z_i = reshape(states(2,3,:),1,length(t));
p_i = [x_i;y_i;z_i];

figure1=figure('Position', [200, 100, 1024/2, 1200]);

%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Plot Pos Vel
%%%%%%%%%%%%%%%%%%%%%%%%%%%

tiledlayout(6,1, 'Padding', 'compact', 'TileSpacing', 'compact')


nexttile
plot(t,x_o,'Linewidth',line_width)
ylabel('x (m)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
set(gca,'XTick',[])
xlim([0,7.5])
grid on


nexttile
plot(t,y_o,'Linewidth',line_width)
ylabel('y (m)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
set(gca,'XTick',[])
xlim([0,7.5])
grid on


nexttile
plot(t,z_o,'Linewidth',line_width)
ylabel('z (m)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
set(gca,'XTick',[])
xlim([0,7.5])
grid on


nexttile
hold on
plot(t,vx_o,'Linewidth',line_width)
plot(t,vx_target_o,'Linewidth',line_width,'LineStyle','--')
ylabel('$v_x$ (m/s)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
set(gca,'XTick',[])
xlim([0,7.5])
grid on



nexttile
hold on
plot(t,vy_o,'Linewidth',line_width)
plot(t,vy_target_o,'Linewidth',line_width,'LineStyle','--')
ylabel('$v_y$ (m/s)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
set(gca,'XTick',[])
xlim([0,7.5])
grid on


nexttile
hold on
plot(t,vz_o,'Linewidth',line_width)
plot(t,vz_target_o,'Linewidth',line_width,'LineStyle','--')
ylabel('$v_z$ (m/s)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
grid on
xlabel( 't (s)','fontsize',font_size,'Interpreter','latex');
xlim([0,7.5])


lg = legend('Simulation','Desired','NumColumns',2);
lg.Layout.Tile = 'North';


%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Plot Attitude Attitude_d
%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure2=figure('Position', [200, 100, 1024/2, 1200]);
tiledlayout(6,1, 'Padding', 'compact', 'TileSpacing', 'compact')

nexttile
plot(t,roll_o,'Linewidth',line_width)
ylabel('$\phi$ (rad)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
set(gca,'XTick',[])
xlim([0,7.5])
grid on


nexttile
plot(t,pitch_o,'Linewidth',line_width)
ylabel('$\theta$ (rad)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
set(gca,'XTick',[])
xlim([0,7.5])
grid on


nexttile
plot(t,yaw_o,'Linewidth',line_width)
ylabel('$\psi$ (rad)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
set(gca,'XTick',[])
xlim([0,7.5])
grid on


nexttile
plot(t,rolld_o,'Linewidth',line_width)
ylabel('$\dot{\phi}$ (rad/s)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
set(gca,'XTick',[])
xlim([0,7.5])
grid on



nexttile
plot(t,pitchd_o,'Linewidth',line_width)
ylabel('$\dot{\theta}$ (rad/s)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
set(gca,'XTick',[])
xlim([0,7.5])
grid on


nexttile
plot(t,yawd_o,'Linewidth',line_width)
ylabel('$\dot{\psi}$ (rad/s)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
grid on
xlabel( 't (s)','fontsize',font_size,'Interpreter','latex');
xlim([0,7.5])


%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Plot Ctrl Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure3=figure('Position', [200, 100, 1024, 600]); 

tiledlayout(4,1, 'Padding', 'compact', 'TileSpacing', 'compact')

nexttile
plot(t,m1_o,'Linewidth',line_width)
ylabel('M1 (rpm)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
grid on
xlim([0,7.5])

nexttile
plot(t,m2_o,'Linewidth',line_width)
ylabel('M2 (rpm)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
grid on
xlim([0,7.5])

nexttile
plot(t,m3_o,'Linewidth',line_width)
ylabel('M3 (rpm)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
grid on
xlim([0,7.5])

nexttile
plot(t,m4_o,'Linewidth',line_width)
ylabel('M4 (rpm)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
grid on
xlim([0,7.5])
xlabel( 't (s)','fontsize',font_size,'Interpreter','latex');


%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Plot Doi and Turn angle
%%%%%%%%%%%%%%%%%%%%%%%%%%%

d_oi = zeros(1,length(t));
turn_angle = zeros(1,length(t));
dir_vector = [1,0,0];

v_o = [vx_o;vy_o;vz_o];

for i = 1 : length(t)
    d_oi(i) = norm(p_o(:,i)-p_i(:,i));
    v_cur_hat = v_o(:,i)/norm(v_o(:,i));
    n = cross(dir_vector,v_cur_hat);
    
    if n(end)<0
        turn_angle_sign = -1;
    else
        turn_angle_sign = 1;
    end
    
    turn_angle(i) = turn_angle_sign*acosd(dot(v_cur_hat,dir_vector));
end

figure
plot(t,d_oi,'Linewidth',line_width)
ylabel('$d_{oi}$ (m)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
grid on
xlim([0,7.5])
xlabel( 't (s)','fontsize',font_size,'Interpreter','latex');

figure
plot(t,turn_angle,'Linewidth',line_width)
ylabel('Turn Angle (deg)','fontsize',font_size,'Interpreter','latex');
set(gca,'XMinorGrid','off','GridLineStyle','-','FontSize',line_size)
grid on
xlim([0,7.5])
xlabel( 't (s)','fontsize',font_size,'Interpreter','latex');
