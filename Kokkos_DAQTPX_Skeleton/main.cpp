#include<Kokkos_Core.hpp>
#include<data.h>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
    if(argc<3)
      printf("Error: expect at least: 'FILENAME N_entries' as arguments\n");
    char * filename = argv[1];
    int N = atoi(argv[2]);
    int team_size = atoi(argv[3]);
    int vec_len = atoi(argv[4]);

    Kokkos::Profiling::pushRegion("init");
    collector_data<Kokkos::HostSpace> collector;
    collector.read_file(filename);    
    /*if(N>0)
      for(int sector = 0; sector<collector.num_sectors; sector++) 
      for(int row = 0; row<collector.num_rows; row++) 
      for(int pad = 0; pad<collector.num_pads(row); pad++) {
      int num_signals = collector.pad_signal_offsets(sector,row,pad+1)-collector.pad_signal_offsets(sector,row,pad);
      int first_signal = collector.pad_signal_offsets(sector,row,pad);
      printf("%i %i %i %i || ",sector,row,pad,num_signals);
      for(int signal = first_signal; signal<first_signal+num_signals; signal++) { 
      printf("%i ",collector.signal_flag(signal));
      printf("%i ",collector.signal_time(signal));
      printf(": ");
      int first_entry = collector.signal_offsets(signal);
      int num_entries = collector.signal_offsets(signal+1) - first_entry;
      for(int e = first_entry; e<first_entry+num_entries; e++)
      printf("%i ",collector.signal_values(e));
      printf("| ");
      }
      printf("\n");
      }
    */
    typedef collector_data<Kokkos::DefaultExecutionSpace::memory_space> t_gpu_collector_data;

#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::View<t_gpu_collector_data*, Kokkos::CudaHostPinnedSpace> events("all_events", N);
#else 
    Kokkos::View<t_gpu_collector_data*, Kokkos::HostSpace> events("all_events", N);
#endif

    for (int i = 0; i < N; ++i) {
      events(i) = t_gpu_collector_data(collector.num_sectors,collector.num_rows,
                                       collector.max_num_pads,collector.total_num_signals,
                                       collector.signal_values.extent(0));
      deep_copy(events(i), collector);
    }
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("real_work");
Kokkos::Timer timer;
    for (int e = 0; e < N; ++e) {
      t_gpu_collector_data data = events(e);
      Kokkos::parallel_for("init_blob_id", data.total_num_signals, KOKKOS_LAMBDA(const int i){
          data.blob_id(i) = i;
        });	
    }

    Kokkos::View<int32_t**, Kokkos::LayoutRight> blob_size("blob_size", N, events(0).num_sectors*events(0).num_rows*300);
    Kokkos::View<int32_t**, Kokkos::LayoutRight> blob_offset("blob_offset", N, events(0).num_sectors*events(0).num_rows*300);
    Kokkos::View<int32_t*> blob_counts("blob_counts", N);
    Kokkos::View<int32_t*>::HostMirror h_blob_counts = Kokkos::create_mirror_view(blob_counts);

    Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>> policy(24*N, team_size, vec_len);
    Kokkos::parallel_for ("sector loop", policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& t) {
	int iEvent  = t.league_rank()/24;
	t_gpu_collector_data data = events(iEvent);

        int iSector = t.league_rank()%24+1;
	Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 1, data.num_rows+1), [&](const int& iRow) {
	    // if (iSector != 1 || iRow != 1) continue;
	    int not_done = 1;
	    while (not_done) {
	      not_done = 0;
	      for (int iIter = 0; iIter < 2; iIter++) {
		Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(t, data.num_pads(iRow)/2), [=](const int& iPad2, int& thead_not_done) {
		    const int iPad = iPad2*2 + iIter;
		    const int num_signals = data.pad_signal_offsets(iSector,iRow,iPad+1)-data.pad_signal_offsets(iSector,iRow,iPad);
		    const int first_signal = data.pad_signal_offsets(iSector,iRow,iPad);

		    int first_nb_signal = data.pad_signal_offsets(iSector,iRow,iPad+1);
		    const int last_nb_signal = data.pad_signal_offsets(iSector,iRow,iPad+2);
          
		    for (int iSignal = first_signal; iSignal < first_signal+num_signals; iSignal++) {
		      const int signal_time_high = data.signal_time(iSignal);
		      const int signal_length = data.signal_offsets(iSignal+1) - data.signal_offsets(iSignal);
		      const int signal_time_low  = signal_time_high - signal_length + 1;
            
		      for (int iSignal_nb = first_nb_signal; iSignal_nb < last_nb_signal; iSignal_nb++) {
			const int signal_nb_time_high = data.signal_time(iSignal_nb);
			const int signal_nb_length = data.signal_offsets(iSignal_nb+1) - data.signal_offsets(iSignal_nb);
			const int signal_nb_time_low  = signal_nb_time_high - signal_nb_length + 1;

			if (signal_nb_time_high >= signal_time_low &&
			    signal_nb_time_low <= signal_time_high) {

			  if (data.blob_id(iSignal) != data.blob_id(iSignal_nb)) {
			    thead_not_done = 1;
			  }

			  if (data.blob_id(iSignal) < data.blob_id(iSignal_nb)) {
			    data.blob_id(iSignal_nb) = data.blob_id(iSignal);
			  } else {
			    data.blob_id(iSignal) = data.blob_id(iSignal_nb);
			  }
			}
		      } // iSignal_nb loop end
		    }   // iSignal loop end
		  }, not_done); // pad loop end
	      }	// Iter loop end
	    } // while done end

	    int first_row_signal = data.pad_signal_offsets(iSector,iRow,0);
	    int last_row_signal = data.pad_signal_offsets(iSector,iRow,data.num_pads(iRow));
	    Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, last_row_signal-first_row_signal), [&](const int iSignal_iter) {
		int iSignal = iSignal_iter + first_row_signal;
		if (data.blob_id(iSignal) == iSignal) {
		  int last_blob_counts = Kokkos::atomic_fetch_add(&blob_counts(iEvent), 1);
		  data.blob_id(iSignal) = -(last_blob_counts + 1);
		}
	      });

	    Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, last_row_signal-first_row_signal), [&](const int iSignal_iter) {
		int iSignal = iSignal_iter + first_row_signal;
		if (data.blob_id(iSignal) >= 0) {
		  int blob_head_id = data.blob_id(iSignal);
		  data.blob_id(iSignal) = data.blob_id(blob_head_id);
		}
		Kokkos::atomic_increment(&blob_size(iEvent, -data.blob_id(iSignal)));
	      });
        
	  });   // row loop end
      });   // sector loop end
  
    Kokkos::Profiling::pushRegion("blob_preparation");
    Kokkos::deep_copy(h_blob_counts, blob_counts);
    for (int iEvent = 0; iEvent < N; ++iEvent) {
      int h_bcounts = h_blob_counts(iEvent);

      Kokkos::Profiling::pushRegion("alloc_blob_data");
      events(iEvent).alloc_blob_data(h_bcounts);
      Kokkos::Profiling::popRegion(); 
  
      Kokkos::parallel_scan("comp_global_offset", h_bcounts, KOKKOS_LAMBDA(const int& iBlob, int& global_offset, bool final){
          global_offset += blob_size(iEvent, iBlob); 
          if (final) blob_offset(iEvent, iBlob) = global_offset;
        });
    }

    Kokkos::Profiling::popRegion();

    Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>> policy2(N, Kokkos::AUTO);
    Kokkos::parallel_for("do_peaks", policy2, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& t){
        int iEvent = t.league_rank();
        t_gpu_collector_data data = events(iEvent);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, blob_counts(iEvent)+1), [&](const int& i) {
            blob_size(iEvent, i) = 0;
          });
       
        t.team_barrier(); 
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, data.total_num_signals), [&](const int& iSignal){
            int id = -data.blob_id(iSignal);
            int myOffset = Kokkos::atomic_fetch_add(&blob_size(iEvent, id), 1);
            int myblob_offset = blob_offset(iEvent, id-1);
            data.blob_signal_map(myOffset+myblob_offset) = iSignal;
          });
        t.team_barrier(); 

        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 1, blob_counts(iEvent)), [&](const int& iBlob){
            int my_blob_offset = blob_offset(iEvent, iBlob-1);
            float cluster_tb = 0, cluster_pad = 0, cluster_ADC = 0;
            for (int i = my_blob_offset; i < my_blob_offset+blob_size(iEvent, iBlob); i++) {
              int iSignal = data.blob_signal_map(i);
              int Signal_trailing_tb = data.signal_time(iSignal);
              int Signal_pad = data.signal_pad(iSignal);
              for (int iADC = data.signal_offsets(iSignal+1)-1; iADC > data.signal_offsets(iSignal)-1; iADC--) {
                cluster_tb += Signal_trailing_tb * data.signal_values(iADC);
                cluster_pad += Signal_pad * data.signal_values(iADC);
                cluster_ADC += data.signal_values(iADC);
                Signal_trailing_tb--;
              }
            }
            data.clusters(iBlob, 0) = cluster_tb / cluster_ADC;
            data.clusters(iBlob, 1) = cluster_pad / cluster_ADC;
            // printf("Blob: %i, size: %i, cluster_tb %f, cluster_pad %f, cluster_ADC %f\n", iBlob, blob_size(iBlob), cluster_tb, cluster_pad, cluster_ADC);
          });
      }); // done peaks
  Kokkos::fence();
  double time = timer.seconds();


  printf("Events: %i Events/s: %lf Time: %lf\n",N,1.0*N/time,time);
  Kokkos::Profiling::popRegion();
 
  for (int iEvent = 0; iEvent < N; ++iEvent) {
	events(iEvent).free_mem();
  }

  } // init


  Kokkos::finalize();
}
