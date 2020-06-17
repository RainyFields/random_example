function p = un_boltzmann_dist(w,theta,pattern)
% size pattern 1*10
% size theta 10*1
ss = pattern*pattern';
temp = w./ss;
p = exp(1/2*sum(temp(:))+sum(theta.*pattern'));
end