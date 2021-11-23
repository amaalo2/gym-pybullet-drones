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

for i = 1 : length(t)
    d_oi(i) = norm(p_o(:,i)-p_i(:,i));
end


grad_doi = gradient(d_oi);


%Find all indices to perform the splice
k=1;
for i = 1 : length(d_oi)
    if grad_doi(i) > 5
        idx(k) = i;
        k = k+1;
    end
end

%Add the endpoints for the splice
idx = [1, idx, length(t)];


%create min plot

k = 1;
for i = 1 : 2 : length(idx)

min_doi(k) = min(d_oi(1,idx(i):idx(i+1)));
k = k+1;
end




figure
plot(t,x_o)
%hold on
figure
plot(t,d_oi)