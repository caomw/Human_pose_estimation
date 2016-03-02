function draw_figure_1(depth)

xmin = min(depth(:,1));
xmax = max(depth(:,1));
ymin = min(depth(:,3));
ymax = max(depth(:,3));
zmin = min(depth(:,2));
zmax =  max(depth(:,2));

figure;
scatter3(depth(:,1), depth(:,3), depth(:,2),ones(size(depth,1),1) * 3, depth(:,3), 'filled');





set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])
set(gca,'ZTickLabel',[])

planeColor = [0.0000 1.0000 1.0000 0.0000];
hold on;
X = [xmin, xmin, xmax, xmax];
Y = [ymin, ymax, ymax, ymin];
Z = [zmin, zmin, zmin, zmin];
fill3(X,Y,Z,planeColor);
xlim([xmin xmax]);
ylim([ymin ymax]);
zlim([zmin zmax]);

end