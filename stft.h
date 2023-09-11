#ifndef STFT__STFT_H_
#define STFT__STFT_H_

#include "kiss_fft/kiss_fft.h"
#include "signal_processing/buffer.h"

// TODO: Add synthesis window ISTFT.
typedef struct StftState {
  int nfft;      // Number of samples in the fft.
  int hop_size;  // Number of samples between each window. For 50% overlap,
                 // hop_size = window_size / 2.
  const float* window;  // Window function. Same length as nfft.
  kiss_fft_cfg cfg;     // kiss_fft configuration.
  DoubleBufferComplexInterleaved* buffer;  // Buffer for input samples.
  kiss_fft_cpx* in;                        // Input signal, length is nfft.
  kiss_fft_cpx* out;  // Output signal, length is (nfft + 1) / 2 since we only
                      // care about the real part. Assuming nfft is even.
} StftState;

// Generate a Hann window of specified length.
void hann_window(float* window, int length);

// Generate a synthesis filter bases on anlysis filter, hop size and nfft.
// The synthesis filter is the inverse of the sum of all overlapping analysis
// windows inside a single frame. This function only work where nfft is a
// multiple of hop_size.
// The synthesis filter is flipped so that the last hop_size samples are the
// first etc.
void calculate_synthesis_window(const float* analysis_filter, int nfft,
                                int hop_size, float* synthesis_filter);

// Initializes the stft state.
// Using a Hann window and hop_size = nfft / 2 gives means you need no synthesis
// window when doing inverse stft.
// The double_dubber is assumed to have a size of nfft.
void stft_init(const float* window, DoubleBufferComplexInterleaved* buffer,
               int nfft, int hop_size, StftState* state);

// Initializes the inverse stft state.
// The double_dubber is assumed to have a size of (M^2 - 1) * hop_size where M
// is nfft/hop_size which is the number of overlaps in one frame.
void stft_inverse_init(const float* window,
                       DoubleBufferComplexInterleaved* buffer, int nfft,
                       int hop_size, StftState* state);

// Update input buffer with hop_size new input samples.
void stft_buffer_update(const float* x_real, const float* x_imag,
                        StftState* state);

// Calculates the short time fourier transform.
void stft(StftState* state);

// Calculates the inverse short time fourier transform.
void stft_inverse(StftState* state, float* x_stft_real, float* x_stft_imag);

// Retrieve the output of the STFT.
void stft_get_out(const StftState* state, float* x_stft_real,
                  float* x_stft_imag);

// Adds together the nfft / hop_size overlapping frames and applies the
// synthesis window.
void stft_inverse_get_out(const StftState* state, float* x_real, float* x_imag);

// Free the memory allocated by the stft state.
void stft_free(StftState* state);

#endif  // STFT__STFT_H_
