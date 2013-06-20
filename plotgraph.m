function plotgraph(Y)

greedy = (Y(1:20,1))';
  mto = (Y(1:20,2))';
qlearning = (Y(1:20,3))';
  rel_qlearning = (Y(1:20,4))';
rlearning = (Y(1:20,5))';
xaxis = zeros(1,20);
for n=1:20
	xaxis(n) = 0.5+(n-1)*0.1;
	end;
plot(xaxis,greedy,'r-',xaxis,mto,'b-',xaxis,qlearning,'g-',xaxis,rel_qlearning,'m-',xaxis,rlearning,'y-');
legend("Greedy","MTO","Qlearning","Relative Qlearning","Rlearning");
print("Graph1.jpg");





