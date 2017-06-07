#include<Kokkos_Core.hpp>
#include<data.h>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
    if(argc<3)
      printf("Error: expect at least: 'FILENAME N_entries' as arguments\n");
    char * filename = argv[1];
    int N = atoi(argv[2]);

    collector_data<Kokkos::HostSpace> collector;
    collector.read_file(filename);    
    if(N>0)
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

    collector_data<Kokkos::DefaultExecutionSpace::memory_space> data(collector.num_sectors,collector.num_rows,
                                                                     collector.max_num_pads,collector.total_num_signals,
                                                                     collector.signal_values.extent(0));
  
    printf("DeepCopy\n");  
    deep_copy(data,collector);
    Kokkos::deep_copy(data.forward_link, -1);
    Kokkos::deep_copy(data.backward_link, -1);
    Kokkos::parallel_for(data.total_num_signals, KOKKOS_LAMBDA(const int i){
        data.blob_id(i) = i;
      });
    Kokkos::View<int32_t*, Kokkos::LayoutLeft> blob_size("blob_size", data.num_sectors*data.num_rows*300);

    int blob_counts = 0;
    
    for (int iSector = 0; iSector < data.num_sectors; iSector++) {
      for (int iRow = 0; iRow < data.num_rows; iRow++) {
        bool not_done = true;
        while (not_done) {
          not_done = false;
          for (int iIter = 0; iIter < 2; iIter++)
            for (int iPad = iIter; iPad < data.num_pads(iRow)-1; iPad += 2) {
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
                    // data.forward_link(iSignal) = iSignal_nb;
                    // data.backward_link(iSignal_nb) = iSignal;

                    if (data.blob_id(iSignal) != data.blob_id(iSignal_nb)) {
                      not_done = true;
                    }

                    if (data.blob_id(iSignal) < data.blob_id(iSignal_nb)) {
                      data.blob_id(iSignal_nb) = data.blob_id(iSignal);
                    } else {
                      data.blob_id(iSignal) = data.blob_id(iSignal_nb);
                    }
                  }
                } // iSignal_nb loop end
              }   // iSignal loop end
            } // pad loop end
        } // while done end

        int first_row_signal = data.pad_signal_offsets(iSector,iRow,0);
        int last_row_signal = data.pad_signal_offsets(iSector,iRow,data.num_pads(iRow));
        for (int iSignal = first_row_signal; iSignal < last_row_signal; iSignal++) {
          if (data.blob_id(iSignal) == iSignal) {
            blob_counts++;
            data.blob_id(iSignal) = -blob_counts;
          }
        }

        for (int iSignal = first_row_signal; iSignal < last_row_signal; iSignal++) {
          if (data.blob_id(iSignal) >= 0) {
            int blob_head_id = data.blob_id(iSignal);
            data.blob_id(iSignal) = data.blob_id(blob_head_id);
          }
          blob_size(-data.blob_id(iSignal))++;
        }
        
      }   // row loop end
    }     // sector loop end

    for (int iBlob = 1; iBlob <= blob_counts; iBlob++) {
      printf("Blob: %i, size: %i\n", iBlob, blob_size(iBlob));
    }
  
  } // init

  Kokkos::finalize();
}