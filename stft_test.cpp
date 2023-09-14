
extern "C" {
#include "stft.h"
}

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(HannWindowTest, HannWindow) {
  std::vector<float> window(10, 0.0f);
  hann_window(window.data(), 10);
  std::vector<float> expected = {
      0,           0.116977781, 0.413175881, 0.75,        0.969846308,
      0.969846308, 0.74999994,  0.413175941, 0.116977841, 0};
  EXPECT_THAT(expected, Pointwise(FloatNear(0.001), window));
}

// Tests that synthesis_filter generates a correct synthesis filter.
TEST(SynthesisFilterTest, Generate) {
  int nfft = 8;
  int hop_size = 4;
  std::vector<float> window(nfft);
  hann_window(window.data(), nfft);
  std::vector<float> synthesis_window(nfft);
  calculate_synthesis_window(window.data(), nfft, hop_size,
                             synthesis_window.data());
  std::vector<float> expected = {1.05209508, 1.25075739, 1.25075739,
                                 1.05209508, 1.05209508, 1.25075739,
                                 1.25075739, 1.05209508};
  EXPECT_THAT(expected, Pointwise(FloatNear(0.001), synthesis_window));
}

TEST(StftTest, InitNormally) {
  int nfft = 8;
  int hop_size = 4;
  std::vector<float> buffer_interleaved(
      4 * nfft);  // The 4 is due to complex numbers and it
                  // being a double buffer (2*2).
  DoubleBufferComplexInterleaved double_buffer =
      double_buffer_complex_interleaved_init(buffer_interleaved.data(), nfft);
  std::vector<float> window(nfft);
  hann_window(window.data(), nfft);
  StftState state;
  stft_init(window.data(), &double_buffer, nfft, hop_size, &state);
  stft_free(&state);
}

TEST(StftTest, InitInverseNormally) {
  int nfft = 8;
  int hop_size = 4;
  int num_overlaps = nfft / hop_size;
  int num_hop_frames = (num_overlaps * num_overlaps + 1) * hop_size;
  std::vector<float> buffer_interleaved(
      4 * num_hop_frames);  // The 4 is due to complex numbers and it
                            // being a double buffer (2*2).
  DoubleBufferComplexInterleaved double_buffer =
      double_buffer_complex_interleaved_init(buffer_interleaved.data(), nfft);
  std::vector<float> window(nfft);
  hann_window(window.data(), nfft);
  StftState state;
  stft_inverse_init(window.data(), &double_buffer, nfft, hop_size, &state);
  stft_free(&state);
}

// Initializes an STFT state and tests if stft_buffer_update is working.
TEST(StftTest, StftStep) {
  int nfft = 8;
  int hop_size = 4;
  std::vector<float> buffer_interleaved(
      4 * nfft);  // The 4 is due to complex numbers and it
                  // being a double buffer (2*2).
  DoubleBufferComplexInterleaved double_buffer =
      double_buffer_complex_interleaved_init(buffer_interleaved.data(), nfft);
  std::vector<float> window(nfft);
  hann_window(window.data(), nfft);
  StftState state;
  stft_init(window.data(), &double_buffer, nfft, hop_size, &state);

  std::vector<float> x_real = {1.0, 1.0, 2.0, 2.0};
  std::vector<float> x_imag = {-1.0, 1.0, 3.0, 4.0};
  stft_buffer_update(x_real.data(), x_imag.data(), &state);
  stft(&state);

  std::vector<float> x_stft_real(nfft);
  std::vector<float> x_stft_imag(nfft);
  stft_get_out(&state, x_stft_real.data(), x_stft_imag.data());

  // Expected values calculated from scipy.fft.fft (v 1.10.0).
  std::vector<float> expected_x_stft_real = {
      1.9382551,  -2.37970257, 1.1852347,   -0.38571914,
      0.71573417, -0.65079689, -0.03728623, -0.38571914};
  std::vector<float> expected_x_stft_imag = {
      0.22554133, 1.32699463, -2.1265102,  1.43842708,
      -0.9969796, 1.32699463, -0.90398926, -0.29047861};
  EXPECT_THAT(expected_x_stft_real, Pointwise(FloatNear(0.001), x_stft_real));
  EXPECT_THAT(expected_x_stft_imag, Pointwise(FloatNear(0.001), x_stft_imag));

  // Step 2.
  std::vector<float> x_real2 = {1.2, -1.0, 0.2, -2.0};
  std::vector<float> x_imag2 = {1.0, -1.0, 3.0, -0.4};
  stft_buffer_update(x_real2.data(), x_imag2.data(), &state);
  stft(&state);
  stft_get_out(&state, x_stft_real.data(), x_stft_imag.data());

  // Expected values calculated from scipy.fft.fft (v 1.10.0).
  std::vector<float> expected_x_stft_real2 = {
      3.87871677, 2.60330852,  -4.34453374, 1.62296651,
      0.92278977, -2.34643895, 4.10535247,  -6.44216136};
  std::vector<float> expected_x_stft_imag2 = {
      6.7279635,   -6.16791828, 0.87591197, 0.4478877,
      -0.02990124, 1.89720958,  -3.7720365, 0.02088326};
  EXPECT_THAT(expected_x_stft_real2, Pointwise(FloatNear(0.001), x_stft_real));
  EXPECT_THAT(expected_x_stft_imag2, Pointwise(FloatNear(0.001), x_stft_imag));

  // Step 3, first time we wrap around the internal double buffer.
  std::vector<float> x_real3 = {-1.2, 1.0, 0.5, -2.0};
  std::vector<float> x_imag3 = {1.0, -1.0, 0.3, 0.4};
  stft_buffer_update(x_real3.data(), x_imag3.data(), &state);
  stft(&state);
  stft_get_out(&state, x_stft_real.data(), x_stft_imag.data());

  // Expected values calculated from scipy.fft.fft (v 1.10.0).
  std::vector<float> expected_x_stft_real3 = {
      -2.40216518, 3.72700366, -1.77628276, -1.38529628,
      0.55376182,  2.10876872, -0.93763917, 0.11184918};
  std::vector<float> expected_x_stft_imag3 = {
      1.66103302, 1.49886944,  -3.26374773, 0.41922341,
      4.0204517,  -3.45608739, 1.38420074,  -2.26394319};
  EXPECT_THAT(expected_x_stft_real3, Pointwise(FloatNear(0.001), x_stft_real));
  EXPECT_THAT(expected_x_stft_imag3, Pointwise(FloatNear(0.001), x_stft_imag));

  stft_free(&state);
}

// Performs STFT on a buffer and then inverse STFT on the output of the STFT.
// Compares that the result of the inverse STFT is the same as the original
// input to STFT.
TEST(StftTest, Reconstruction) {
  int nfft = 8;
  int hop_size = 4;
  int num_overlaps = nfft / hop_size;
  int num_hop_frames = (num_overlaps * num_overlaps - 1) * hop_size;

  // Initialize STFT.
  std::vector<float> input_buffer(4 *
                                  nfft);  // The 4 is due to complex numbers and
                                          // it being a double buffer (2*2).
  DoubleBufferComplexInterleaved input_double_buffer =
      double_buffer_complex_interleaved_init(input_buffer.data(), nfft);
  std::vector<float> window(nfft);
  hann_window(window.data(), nfft);
  StftState state;
  stft_init(window.data(), &input_double_buffer, nfft, hop_size, &state);

  // Initialize inverse STFT.
  std::vector<float> output_buffer(4 * num_hop_frames);
  DoubleBufferComplexInterleaved output_double_buffer =
      double_buffer_complex_interleaved_init(output_buffer.data(),
                                             num_hop_frames);
  std::vector<float> synthesis_window(nfft);
  calculate_synthesis_window(window.data(), nfft, hop_size,
                             synthesis_window.data());
  StftState state_inverse;
  stft_inverse_init(synthesis_window.data(), &output_double_buffer, nfft,
                    hop_size, &state_inverse);

  // Step 1.
  std::vector<float> x_real = {1.0, 1.0, 2.0, 2.0};
  std::vector<float> x_imag = {-1.0, 1.0, 3.0, 4.0};
  stft_buffer_update(x_real.data(), x_imag.data(), &state);
  stft(&state);
  std::vector<float> x_stft_real(nfft);
  std::vector<float> x_stft_imag(nfft);
  stft_get_out(&state, x_stft_real.data(), x_stft_imag.data());
  stft_inverse(&state_inverse, x_stft_real.data(), x_stft_imag.data());

  // Step 2.
  std::vector<float> x_real2 = {1.2, -1.0, 0.2, -2.0};
  std::vector<float> x_imag2 = {1.0, -1.0, 3.0, -0.4};
  stft_buffer_update(x_real2.data(), x_imag2.data(), &state);
  stft(&state);
  std::vector<float> x_stft_real2(nfft);
  std::vector<float> x_stft_imag2(nfft);
  stft_get_out(&state, x_stft_real2.data(), x_stft_imag2.data());
  stft_inverse(&state_inverse, x_stft_real2.data(), x_stft_imag2.data());
  std::vector<float> x_real_hat(hop_size);
  std::vector<float> x_imag_hat(hop_size);
  stft_inverse_get_out(&state_inverse, x_real_hat.data(), x_imag_hat.data());
  EXPECT_THAT(x_real, Pointwise(FloatNear(0.001), x_real_hat));
  EXPECT_THAT(x_imag, Pointwise(FloatNear(0.001), x_imag_hat));

  // Step 3, first time we wrap around the internal double buffer.
  std::vector<float> x_real3 = {-1.2, 1.0, 0.5, -2.0};
  std::vector<float> x_imag3 = {1.0, -1.0, 0.3, 0.4};
  stft_buffer_update(x_real3.data(), x_imag3.data(), &state);
  stft(&state);
  std::vector<float> x_stft_real3(nfft);
  std::vector<float> x_stft_imag3(nfft);
  stft_get_out(&state, x_stft_real3.data(), x_stft_imag3.data());
  stft_inverse(&state_inverse, x_stft_real3.data(), x_stft_imag3.data());
  std::vector<float> x_real2_hat(hop_size);
  std::vector<float> x_imag2_hat(hop_size);
  stft_inverse_get_out(&state_inverse, x_real2_hat.data(), x_imag2_hat.data());
  EXPECT_THAT(x_real2, Pointwise(FloatNear(0.001), x_real2_hat));
  EXPECT_THAT(x_imag2, Pointwise(FloatNear(0.001), x_imag2_hat));

  stft_free(&state);
}
}  // namespace
