import re
import os

path = '/opt/FasterLivePortrait/src/pipelines/faster_live_portrait_pipeline.py'
content = open(path).read()

# 1. Basic Ratios
telemetry_run = r'''
        print(f"TELEMETRY: EYE_RATIO_DRIVING: {input_eye_ratio.flatten().tolist()}")
        print(f"TELEMETRY: LIP_RATIO_DRIVING: {input_lip_ratio.flatten().tolist()}")
        c_s_eyes = self.calc_eye_close_ratio(source_lmk[None])
        print(f"TELEMETRY: EYE_RATIO_SOURCE: {c_s_eyes.flatten().tolist()}")
        c_s_lip = self.calc_lip_close_ratio(source_lmk[None])
        print(f"TELEMETRY: LIP_RATIO_SOURCE: {c_s_lip.flatten().tolist()}")
'''
content = content.replace('def _run(self, src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, input_eye_ratio, input_lip_ratio, I_p_pstbk, **kwargs):', 
                          'def _run(self, src_info, x_d_i_info, x_d_0_info, R_d_i, R_d_0, realtime, input_eye_ratio, input_lip_ratio, I_p_pstbk, **kwargs):' + telemetry_run)

# 2. Combined and Feat stats
content = content.replace("combined_eye_ratio_tensor = self.calc_combined_eye_ratio(c_d_eyes_i,", 
                          "combined_eye_ratio_tensor = self.calc_combined_eye_ratio(c_d_eyes_i, \n                        print(f'TELEMETRY: EYE_COMBINED: {combined_eye_ratio_tensor.flatten().tolist()}'),")

# Since we want to print AFTER it's defined, let's be more precise
content = re.sub(r'(combined_eye_ratio_tensor = self\.calc_combined_eye_ratio\(.*?\))', 
                 r'\1\n                        print(f"TELEMETRY: EYE_COMBINED: {combined_eye_ratio_tensor.flatten().tolist()}")', content, flags=re.DOTALL)

content = re.sub(r'(feat_eye = concat_feat\(kp_source, eye_close_ratio\))',
                 r'\1\n        print(f"TELEMETRY: FEAT_EYE mean={feat_eye.mean().item():.6f}, std={feat_eye.std().item():.6f}")', content)

# 3. Final KP
content = content.replace('out_crop = self.model_dict["warping_spade"].predict(f_s, x_s, x_d_i_new)', 
                          'print(f"TELEMETRY: KP_FINAL mean={x_d_i_new.mean().item():.6f}, std={x_d_i_new.std().item():.6f}")\n            out_crop = self.model_dict["warping_spade"].predict(f_s, x_s, x_d_i_new)')

with open(path, 'w') as f:
    f.write(content)
