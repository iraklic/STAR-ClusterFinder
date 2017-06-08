

// Example: 5 signals : 2 signals for (0,1,1) , 1 signal for (0,1,3), 2 signals (0,1,5)
// signal_length: 3, 2, 4, 5, 6
// signal_time: 311, 313, 315 , 515, 715
// signal_offsets: 0, 3, 5, 9, 14, 20, 26
// pad_signal_offsets: (0,0,0)=0...(0,1,1)=0, (0,1,2)=2, (0,1,3)=2, (0,1,4)=3, (0,1,5)=3, 
// getting num_signals for specifc pad: pad_signal_offsets(segm,row,pad+1)-pad_signal_offsets(segm,row,pad)
// getting lenght of signal k in pad: signal_length(pad_signal_offsets(segm,row,pad)+k);
// getting time of signal k in pad: signal_time(pad_signal_offsets(segm,row,pad)+k);
// getting value j of signal k in pad: signal_values(signal_offsets(pad_signal_offsets(sector,row,pad)+k)+j);

template<class MemorySpace>
struct collector_data {
  Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace> num_pads;
  Kokkos::View<int32_t***, Kokkos::LayoutRight, MemorySpace> pad_signal_offsets;
  Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace> signal_offsets;
  Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace> signal_flag;
  Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace> signal_time;
  Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace> signal_pad;
  Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace> signal_values;
  Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace> blob_id;
  Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace> blob_signal_map;
  Kokkos::View<int32_t*[2], Kokkos::LayoutLeft, MemorySpace> clusters;
  
  int num_sectors;
  int num_rows;
  int max_num_pads;
  int total_num_signals;
  
  KOKKOS_INLINE_FUNCTION
  collector_data() {
    num_sectors = 24;
    num_rows = 45;
    max_num_pads = 182;
  }
 
  collector_data(int num_sectors_, int num_rows_, int max_num_pads_, int total_num_signals_, int num_entries) {
    num_sectors = num_sectors_;
    num_rows = num_rows_;
    max_num_pads = max_num_pads_;
    total_num_signals = total_num_signals_;

    num_pads = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>("collector_data::num_pads",num_rows+1);
    pad_signal_offsets = Kokkos::View<int32_t***, Kokkos::LayoutRight, MemorySpace>("collector_data::pad_offsets",num_sectors+1,num_rows+1,max_num_pads+1+1);
    signal_offsets = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>("data_collector::signal_offsets",total_num_signals+1);
    signal_time = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>("data_collector::signal_time",total_num_signals);
    signal_pad = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>("data_collector::signal_pad",total_num_signals);
    signal_flag = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>("data_collector::signal_flag",total_num_signals);
    signal_values = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>("data_collector::signal_values",num_entries);
    blob_id = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>("data_collector::blob_id",total_num_signals);
    blob_signal_map = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>("data_collector::blob_signal_map", total_num_signals+1);
  }

  void free_mem() {
    num_pads = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>();
    pad_signal_offsets = Kokkos::View<int32_t***, Kokkos::LayoutRight, MemorySpace>();
    signal_offsets = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>();
    signal_time = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>();
    signal_pad = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>();
    signal_flag = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>();
    signal_values = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>();
    blob_id = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>();
    blob_signal_map = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>();
  }

  void alloc_blob_data(const int n_blobs) {
    // clusters = Kokkos::View<int32_t*[2], Kokkos::LayoutLeft, MemorySpace>(Kokkos::ViewAllocateWithoutInitializing("data_collector::clusters"), n_blobs);

    clusters = Kokkos::View<int32_t*[2], Kokkos::LayoutLeft, MemorySpace>("data_collector::clusters", n_blobs);
  }

  void read_file(char* filename) {
    Kokkos::View<int32_t***, Kokkos::LayoutRight, MemorySpace> pad_signal_count = Kokkos::View<int32_t***, Kokkos::LayoutRight, MemorySpace>
                   ("collector_data::pad_offsets",num_sectors+1,num_rows+1,max_num_pads+1+1);
    num_pads = Kokkos::View<int32_t*, MemorySpace>("collector_data::num_pads",num_rows+1);
    pad_signal_offsets = Kokkos::View<int32_t***, Kokkos::LayoutRight, MemorySpace>("collector_data::pad_offsets",num_sectors+1,num_rows+1,max_num_pads+1+1);
    init_num_pads();
    FILE* input = fopen(filename,"r");
    int num_lines = 36245;
    // Figure out number of signals per pad
    for(int c=0; c<num_lines; c++) {
      int sector, row, pad;
      int num_signals=0;
      fscanf(input,"%i %i %i",&sector,&row,&pad);
      int signal_len;
      fscanf(input,"%i",&signal_len);
          if((sector == 6) && (row == 7)) printf("TEST1 %i %i %i\n",num_signals,signal_len,pad_signal_offsets(sector,row,pad));
      while(signal_len>0) {
        num_signals++;
        int tmp;
        for(int s = 0; s<signal_len+2; s++) fscanf(input,"%i",&tmp);
        fscanf(input,"%i",&signal_len);
      }
      pad_signal_count(sector,row,pad) += num_signals;
    }
    fclose(input);

    int offset=0;
    for(int sector=0; sector<num_sectors+1; sector++) {
      for(int row=0; row<num_rows+1; row++) {
        for(int pad=0; pad<num_pads(row)+2; pad++) {
          int count = pad_signal_count(sector,row,pad);
          pad_signal_offsets(sector,row,pad) = offset;
          offset+=count;
        }
        pad_signal_offsets(sector,row,num_pads(row)) = offset;
      }
    }

    total_num_signals = offset;
    signal_offsets = Kokkos::View<int32_t*,MemorySpace>("data_collector::signal_offsets",total_num_signals+1);
    signal_time = Kokkos::View<int32_t*,MemorySpace>("data_collector::signal_time",total_num_signals);
    signal_pad = Kokkos::View<int32_t*,MemorySpace>("data_collector::signal_pad",total_num_signals);
    signal_flag = Kokkos::View<int32_t*,MemorySpace>("data_collector::signal_flag",total_num_signals);
    blob_id = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>("data_collector::blob_id",total_num_signals);
    blob_signal_map = Kokkos::View<int32_t*, Kokkos::LayoutLeft, MemorySpace>("data_collector::blob_signal_map",total_num_signals+1);

    Kokkos::deep_copy(pad_signal_count,0);
    // Read signal lengths, time and flag ; lengths are read into offset array doing scan later
    input = fopen(filename,"r");
    for(int c=0; c<num_lines; c++) {
      int sector, row, pad;
      fscanf(input,"%i %i %i",&sector,&row,&pad);
      int signal_len;
      int first_signal_offset = pad_signal_offsets(sector,row,pad);
      fscanf(input,"%i",&signal_len);
      while(signal_len>0) {
        int signal_count = pad_signal_count(sector,row,pad);
        signal_offsets(first_signal_offset+signal_count) = signal_len;
        fscanf(input,"%i",&signal_flag(first_signal_offset+signal_count));
        fscanf(input,"%i",&signal_time(first_signal_offset+signal_count));
        signal_pad(first_signal_offset+signal_count) = pad;
        pad_signal_count(sector,row,pad)++;
        int tmp;
        for(int s = 0; s<signal_len; s++) fscanf(input,"%i",&tmp);
        fscanf(input,"%i",&signal_len);
      }
    }
    fclose(input);

    // Do scan on signal_offsets
/*    Kokkos::parallel_scan("DataCollector::scan_signal_offsets", total_num_signals, 
      KOKKOS_LAMBDA (const int32_t& i, int32_t& lsum, bool final) {
      int32_t length = signal_offsets(i);
      if(final) signal_offsets(i) = lsum;
      lsum += length;
    });
*/

    // Serial
    int count = 0;
    for(int i=0;i<total_num_signals+1; i++) {
      int32_t length = signal_offsets(i);
      signal_offsets(i) = count;
      count += length;
    }

    signal_values = Kokkos::View<int32_t*, MemorySpace>("data_collector::signal_values",count);
    input = fopen(filename,"r");
    for(int c=0; c<num_lines; c++) {
      int sector, row, pad;
      int num_signals=0;
      fscanf(input,"%i %i %i",&sector,&row,&pad);
      int signal_len;
      int first_signal_offset = pad_signal_offsets(sector,row,pad);
      fscanf(input,"%i",&signal_len);
      while(signal_len>0) {
        //signal_offset(first_signal_offset+num_signals) = signal_len;
        int tmp;
        fscanf(input,"%i",&tmp);
        fscanf(input,"%i",&tmp);
        int signal_data_offset = signal_offsets(first_signal_offset+num_signals);
        for(int s = 0; s<signal_len; s++) fscanf(input,"%i",&signal_values(signal_data_offset+s));
        fscanf(input,"%i",&signal_len);
        num_signals++;
      }
    }
    fclose(input);
  }

  void init_num_pads() {
    num_pads(0) = 0;
    num_pads(1) = 88;
    num_pads(2) = 96;
    num_pads(3) = 104;
    num_pads(4) = 112;
    num_pads(5) = 118;
    num_pads(6) = 126;
    num_pads(7) = 134;
    num_pads(8) = 142;
    num_pads(9) = 150;
    num_pads(10) = 158;
    num_pads(11) = 166;
    num_pads(12) = 174;
    num_pads(13) = 182;
    num_pads(14) = 98;
    num_pads(15) = 100;
    num_pads(16) = 102;
    num_pads(17) = 104;
    num_pads(18) = 106;
    num_pads(19) = 106;
    num_pads(20) = 108;
    num_pads(21) = 110;
    num_pads(22) = 112;
    num_pads(23) = 112;
    num_pads(24) = 114;
    num_pads(25) = 116;
    num_pads(26) = 118;
    num_pads(27) = 120;
    num_pads(28) = 122;
    num_pads(29) = 122;
    num_pads(30) = 124;
    num_pads(31) = 126;
    num_pads(32) = 128;
    num_pads(33) = 128;
    num_pads(34) = 130;
    num_pads(35) = 132;
    num_pads(36) = 134;
    num_pads(37) = 136;
    num_pads(38) = 138;
    num_pads(39) = 138;
    num_pads(40) = 140;
    num_pads(41) = 142;
    num_pads(42) = 144;
    num_pads(43) = 144;
    num_pads(44) = 144;
    num_pads(45) = 144;
  }
};

template<class MemorySpace1, class MemorySpace2>
void deep_copy(collector_data<MemorySpace1>& d1, collector_data<MemorySpace2>& d2) {
  Kokkos::deep_copy(d1.num_pads,d2.num_pads);
  Kokkos::deep_copy(d1.pad_signal_offsets,d2.pad_signal_offsets);
  Kokkos::deep_copy(d1.signal_values,d2.signal_values);
  Kokkos::deep_copy(d1.signal_flag,d2.signal_flag);
  Kokkos::deep_copy(d1.signal_offsets,d2.signal_offsets);
  Kokkos::deep_copy(d1.signal_time,d2.signal_time);
  Kokkos::deep_copy(d1.signal_pad,d2.signal_pad);
  Kokkos::deep_copy(d1.blob_id,d2.blob_id);
}
