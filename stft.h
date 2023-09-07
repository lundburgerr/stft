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

// Initializes the stft state.
// Using a Hann window and hop_size = nfft / 2 gives means you need no synthesis
// window when doing inverse stft.
// The double_dubber is assumed to have a size of nfft.
void stft_init(const float* window, DoubleBufferComplexInterleaved* buffer,
               int nfft, int hop_size, StftState* state);

// Update input buffer with hop_size new input samples.
void stft_buffer_update(const float* x_real, const float* x_imag,
                        StftState* state);

// Calculates the short time fourier transform.
void stft(StftState* state);

// Retrieve the output of the STFT.
void stft_get_out(const StftState* state, float* x_stft_real,
                  float* x_stft_imag);

// Free the memory allocated by the stft state.
void stft_free(StftState* state);

#endif  // STFT__STFT_H_
