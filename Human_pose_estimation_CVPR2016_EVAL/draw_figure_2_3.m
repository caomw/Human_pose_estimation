function draw_figure_2_3(sample, cluster) %only for 2-nearest joint

figure;

original_coordinate = sample(:,1:3);
offset_1 = sample(:,4:6);
offset_2 = sample(:,7:9);
cluster_1 = cluster(1,1:3);
cluster_2 = cluster(1,4:6);
cluster_3 = cluster(2,1:3);
cluster_4 = cluster(2,4:6);

%scatter3(original_coordinate(:,1), original_coordinate(:,3), original_coordinate(:,2),ones(size(original_coordinate,1),1) * 10, 'rs', 'filled');
scatter3(offset_1(:,1), offset_1(:,3), offset_1(:,2),ones(size(offset_1,1),1) * 10, 'rs', 'filled');
hold on;
scatter3(offset_2(:,1), offset_2(:,3), offset_2(:,2),ones(size(offset_2,1),1) * 10, 'bs', 'filled');
hold on;
scatter3(cluster_1(:,1),cluster_1(:,3), cluster_1(:,2),ones(size(cluster_1,1),1) * 200, 'gs', 'filled');
hold on;
scatter3(cluster_2(:,1),cluster_2(:,3), cluster_2(:,2),ones(size(cluster_2,1),1) * 200, 'gs', 'filled');
hold on;
scatter3(cluster_3(:,1),cluster_3(:,3), cluster_3(:,2),ones(size(cluster_3,1),1) * 200, 'gs', 'filled');
hold on;
scatter3(cluster_4(:,1),cluster_4(:,3), cluster_4(:,2),ones(size(cluster_4,1),1) * 200, 'gs', 'filled');

colormap(gray)
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])
set(gca,'ZTickLabel',[])

end