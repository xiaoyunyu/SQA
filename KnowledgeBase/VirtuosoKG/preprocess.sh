./bin/virtuoso-t +foreground +configfile var/lib/virtuoso/db/virtuoso.ini & 
serverPID=$!
sleep 10
# ./bin/isql 1111 dba dba exec="ld_dir_all('/home/yxy/data', '*', 'fb:');"
./bin/isql 1111 dba dba exec="ld_add('/home/yxy/data/FB5M.core.txt','fb:');"
# SELECT * FROM DB.DBA.LOAD_LIST; 
# delete from db.dba.load_list;
pids=()
for i in `seq 1 4`; do
	./bin/isql 1111 dba dba exec="rdf_loader_run();" &
   pids+=($!)
done
for pid in ${pids[@]}; do
     wait $pid
done
