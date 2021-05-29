function rslt = bar4celllist(celllist)
    l = unique(celllist);
    rslt = [];
    for i = 1:length(l)
        rslt(i).name = l{i};
        rslt(i).count = sum(strcmp(celllist,l{i}));
    end
end