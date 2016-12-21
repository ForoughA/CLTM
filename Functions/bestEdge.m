clear
figure;
for i=1:30
   load(sprintf('edgeParamsFullTwitter/edgeOutCond%d.mat',i));
   J(i) = eLL(end);
    plot(eParams);
    hold on
end

