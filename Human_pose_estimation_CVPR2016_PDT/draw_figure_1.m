function draw_figure_1(depth, estimated)

figure;

scatter3(depth(:,1), depth(:,3), depth(:,2),ones(size(depth,1),1) * 3, depth(:,3), 'filled');
hold on;
scatter3(estimated(:,1), estimated(:,3), estimated(:,2),ones(size(estimated,1),1) * 50, 'rs', 'filled');

colormap(gray)
set(gca,'XTick',[])
set(gca,'YTick',[])
set(gca,'ZTick',[])
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])
set(gca,'ZTickLabel',[])

end