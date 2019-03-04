benchmarkGtDir = 'MOT16/train/';
seqmap = 'c5-train.txt';
% res = 'res/MOT16/sort_tracker_1/';
% res = 'res/MOT16/sort_tracker/Darknet_nms_thres_0.7_conf_thres_0.6_sort_tracker_max_age_1_min_hits_3/';
% res = 'res/MOT16/iou_tracker_from_gt_det/';
res = 'res/MOT16/sort_tracker/MOT16_gt__sort_tracker_max_age_1_min_hits_3/';
[allMets, metsBenchmark] = evaluateTracking(seqmap, res, benchmarkGtDir, 'MOT16');

%printf(allMets)
%disp(allMets.m)
%print(metsBenchmark)

sequenceListFile = fullfile('seqmaps',seqmap);
allSequences = parseSequences2(sequenceListFile);

metricsInfo.names.long = {'Recall','Precision','False Alarm Rate', ...
    'GT Tracks','Mostly Tracked','Partially Tracked','Mostly Lost', ...
    'False Positives', 'False Negatives', 'ID Switches', 'Fragmentations', ...
    'MOTA','MOTP', 'MOTA Log'};

metricsInfo.names.short = {'Rcll','Prcn','FAR', ...
    'GT','MT','PT','ML', ...
    'FP', 'FN', 'IDs', 'FM', ...
    'MOTA','MOTP', 'MOTAL'};

metricsInfo.widths.long = [4 9 16 9 14 17 11 15 15 11 14 5 5 8];
metricsInfo.widths.short = [4 5 5 4 4 4 4 6 6 5 5 5 5 5];

metricsInfo.format.long = {'.1f','.1f','.2f', ...
    'i','i','i','i', ...
    'i','i','i','i', ...
    '.1f','.1f','.1f'};

metricsInfo.format.short=metricsInfo.format.long;  
fprintf("\n")
for ind = 1:length(allSequences)
   printMetrics(allMets(ind).m, metricsInfo, 0)
end
printMetrics(metsBenchmark, metricsInfo, 0)