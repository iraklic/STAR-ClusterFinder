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
  }
  Kokkos::finalize();
}
