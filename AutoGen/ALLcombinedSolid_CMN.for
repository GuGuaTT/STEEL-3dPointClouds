      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1RPL,DDSDDT,DRPLDE,DRPLDT,
     2STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC)
C
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME
C
      DIMENSION STRESS(NTENS),STATEV(NSTATV),
     1DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS),
     2STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1),
     3PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3),
     4JSTEP(4)
C
	 IF (CMNAME.EQ. 'M-BEAM') THEN
C
		CALL UMAT1(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1	RPL,DDSDDT,DRPLDE,DRPLDT,
     2	STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3	NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4	CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC)
C
	 ELSE IF (CMNAME.EQ. 'M-SOLID') THEN
C
		CALL UMAT2(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1	RPL,DDSDDT,DRPLDE,DRPLDT,
     2	STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3	NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4	CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC)
C
	 END IF
C
	 RETURN
	 END
C
      SUBROUTINE UMAT1(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1RPL,DDSDDT,DRPLDE,DRPLDT,
     2STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC)
      ! Note that IMPLICIT definition is active
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME
C
      DIMENSION STRESS(NTENS),STATEV(NSTATV),
     1DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS),
     2STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1),
     3PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3),
     4JSTEP(4)
C
      ! Variables defined in the subroutine
      INTEGER :: i, j, n_backstresses, it_num, is_converged
      ! Material properties
      REAL(8) :: elastic_modulus, sy_0, Q, b, c_k, gamma_k,
     1ep_eq_init, alpha_init, D, a
      ! Used for intermediate calculations
      REAL(8) :: alpha, sy, sigma, ep_eq, e_p, phi, aux, dit, dep,
     1A_term,  yield_radius, iso_Q, iso_D, e_p_total
      ! Vectors
      REAL(8), DIMENSION(:, :), ALLOCATABLE :: chab_coef
      REAL(8), DIMENSION(:), ALLOCATABLE :: alpha_k, alpha_k_init
      ! Parameters
      INTEGER :: N_BASIC_PROPS, TERM_PER_BACK, MAX_ITERATIONS,
     1DEBUG_ON, n_mat_props
      REAL(8) :: TOL, ONE, TWO, ZERO
      PARAMETER(TOL=1.0D-6,
     1N_BASIC_PROPS=6, TERM_PER_BACK=2, MAX_ITERATIONS=1000,
     2ONE=1.0D0, TWO=2.0D0, ZERO=0.D0)
C-----------------------------------------------------------------------C
C     
      ! Start of subroutine
C
C-----------------------------------------------------------------------C
      ! Set the properties of the material model
      n_is_props = 8
      n_backstresses = (nprops - N_BASIC_PROPS - n_is_props) / TERM_PER_BACK
      IF (n_backstresses .EQ. 0) THEN
        PRINT *, "No backstresses defined, exiting!"
        CALL XIT  ! Exit from analysis command in Abaqus
      END IF
      ALLOCATE(chab_coef(n_backstresses, 2))
      ALLOCATE(alpha_k(n_backstresses))
      ALLOCATE(alpha_k_init(n_backstresses))
C      
      elastic_modulus = props(1)
      sy_0 = props(2)
      Q = props(3)
      b = props(4)
      D = props(5)
      a = props(6)
C-----------------------------------------------------------------------C
C     
      ! Add initial stress
C
C-----------------------------------------------------------------------C
      ! Add initial stress when total time = 0
      IF (time(2) == 0.) THEN
        n_mat_props = N_BASIC_PROPS + TERM_PER_BACK * n_backstresses + 1
        ! kspt (section point) = integration point on cross-section
        stress(1) = stress(1) + resid_npt(kspt, props(n_mat_props:nprops))
      END IF
C-----------------------------------------------------------------------C
C     
      ! Elastic trial step
C
C-----------------------------------------------------------------------C
      sigma = stress(1) + elastic_modulus * dstran(1)
C
      ! Determine isotropic component of hardening
      ep_eq = statev(1)  ! 1st state variable assumed to be equivalent plastic strain
      ep_eq_init = ep_eq
      sy = sy_0 + Q * (ONE - EXP(-b * ep_eq)) -
     1D * (ONE - EXP(-a * ep_eq))
C      
      ! Determine kinematic component of hardening
      alpha = ZERO
      DO i = 1, n_backstresses  ! c and gamma assumed to start at 7th entry
        chab_coef(i, 1) = props(N_BASIC_PROPS - 1 + 2 * i)
        chab_coef(i, 2) = props(N_BASIC_PROPS + 2 * i)
        alpha_k(i) = statev(1 + i)  ! alpha_k assumed to be 2nd, ..., state variables (as many as backstresses)
        alpha_k_init(i) = statev(i + 1)
        alpha = alpha + alpha_k(i)
      END DO
      yield_radius = sigma - alpha
      phi = yield_radius ** 2 - sy ** 2
C-----------------------------------------------------------------------C
C     
      ! Return mapping
C
C-----------------------------------------------------------------------C
      is_converged = 1
      IF (phi .GT. TOL) THEN
        is_converged = 0
      END IF
      it_num = 0
      e_p_total = 0.d0
      DO WHILE (is_converged .EQ. 0 .AND. it_num .LT. MAX_ITERATIONS)
        it_num = it_num + 1
C
        ! Determine the plastic strain increment
        aux = elastic_modulus
        DO i = 1, n_backstresses
          aux = aux + chab_coef(i, 1) -
     1    SIGN(ONE, yield_radius) * chab_coef(i, 2) * alpha_k(i)
        END DO
C
      dit = TWO * yield_radius * aux +
     1 SIGN(ONE, yield_radius) * TWO * sy * Q * b * EXP(-b * ep_eq) -
     2 SIGN(ONE, yield_radius) * TWO * sy * D * a * EXP(-a * ep_eq)
      dep = phi / dit
C 
      ! Prevents newton step from overshooting
      IF (ABS(dep) > ABS(sigma / elastic_modulus)) THEN
        dep = SIGN(ONE, dep) * 0.95D0 *
     1  ABS(sigma / elastic_modulus)
      END IF
C-----------------------------------------------------------------------C
C     
      ! Update variables
C
C-----------------------------------------------------------------------C
      e_p_total = e_p_total + dep
      ep_eq = ep_eq_init + ABS(e_p_total)
      sigma = sigma - elastic_modulus * dep
      iso_Q = Q * (ONE - EXP(-b * ep_eq))
      iso_D = D * (ONE - EXP(-a * ep_eq))
      sy = sy_0 + iso_Q - iso_D
C            
      DO i = 1, n_backstresses
        c_k = chab_coef(i, 1)
        gamma_k = chab_coef(i, 2)
        alpha_k(i) = SIGN(ONE, yield_radius) * c_k / gamma_k -
     1  (SIGN(ONE, yield_radius) * c_k / gamma_k - alpha_k_init(i)) *
     2  EXP(-gamma_k * (ep_eq - ep_eq_init))
      END DO
      alpha = SUM(alpha_k(:))  ! don't put in the loop since will change the SIGN
C            
      yield_radius = sigma - alpha
      phi = yield_radius ** 2 - sy ** 2
C
      ! Check convergence
      IF (ABS(phi) .LT. TOL) THEN
        is_converged = 1
      END IF
      END DO
C      
      ! Set the stress and tangent stiffness (Jacobian)
      DO j = 1, ntens
        DO i = 1, ntens
          ddsdde(i, j) = 0.
        END DO
      END DO
      ! Condition of plastic loading is determined by whether or not iterations were required
      IF (it_num .EQ. 0) THEN
        ddsdde(1, 1) = elastic_modulus
      ELSE
        A_term =  b * (Q - iso_Q) - a * (D - iso_D)
        DO i = 1, n_backstresses
          c_k = chab_coef(i, 1)
          gamma_k = chab_coef(i, 2)
          A_term = A_term +
     1    gamma_k * (c_k/gamma_k-SIGN(ONE, yield_radius)*alpha_k(i))
        END DO
        ddsdde(1, 1) = (elastic_modulus * A_term) /
     1  (elastic_modulus + A_term)
      END IF
      stress(1) = sigma
C
      ! Update the state variables
      statev(1) = ep_eq
      DO i = 1, n_backstresses
        statev(1 + i) = alpha_k(i)
      END DO
C      
      IF (it_num .EQ. MAX_ITERATIONS) THEN
        PRINT *, "WARNING: Return mapping in integration point ", npt,
     1  " of element ", noel, "at section ", KSPT, " did not converge."
        PRINT *, "Reducing time increment to 1/4 of current value."
        PNEWDT = 0.25
      END IF
C
      RETURN
C-----------------------------------------------------------------------C
C     
      ! Initial stress functions
C
C-----------------------------------------------------------------------C
      contains
C
      FUNCTION resid_parab(loc, pm1, pm2) result(rsi)
        ! Returns the value on the residual stress parabola.
        real(8), intent(in) :: loc, pm1, pm2
        real(8) :: rsi
        rsi = pm1 + pm2 * loc ** 2.
      END function
C
      FUNCTION resid_npt(npt_i, geom_params) result(rs)
        ! Returns the residual stress value at the integration point.
        integer, intent(in) :: npt_i
        real(8), intent(in) :: geom_params(:)
        real(8) :: rs
        ! Internal
        real(8) :: d, bf, tf, tw, div_l, xa, a_pm, b_pm, c_pm, d_pm
        integer :: npt_flange, npt_web, web_npt_min, web_npt_max
C
        ! Extract geom
        d = geom_params(1)
        bf = geom_params(2)
        tf = geom_params(3)
        tw = geom_params(4)
C
        ! Extract int pt info
        a_pm = geom_params(5)
        c_pm = geom_params(6)
        npt_flange = geom_params(7)
        npt_web = geom_params(8)
        web_npt_min = npt_flange + 1
        web_npt_max = npt_flange + npt_web - 2
C
        ! Determine the parabolic coefficients
        d_pm = 4. * (a_pm - c_pm) / (d - tf) ** 2.
        b_pm = - (2. * tf * bf * a_pm + tw * (d - 2. * tf) * c_pm + tw * d_pm / 12. * 
     1  (d - 2. * tf) ** 3.) / (2. * tf * bf ** 3. / 12.)
C
        ! Bottom flange
        IF (npt_i < web_npt_min) THEN
C
          div_l = bf / (npt_flange - 1)
          xa = -bf / 2. + div_l * (npt_i - 1)
C
          ! Compute the stress
          rs = resid_parab(xa, a_pm, b_pm)
C
        ! Top flange
        ELSE IF (npt_i > web_npt_max) THEN
C
          div_l = bf / (npt_flange - 1)
          xa = -bf / 2. + div_l * (npt_i - web_npt_max - 1)
C
          ! Compute the stress
          rs = resid_parab(xa, a_pm, b_pm)
C
        ! Web
        ELSE
C
          div_l = (d - tf) / (npt_web - 1)
          xa = -(d - tf) / 2. + div_l * (npt_i - web_npt_min + 1)
C
          ! Compute the stress
          rs = resid_parab(xa, c_pm, d_pm)
C
        END IF
C
      END function
      END
C
      SUBROUTINE UMAT2(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1RPL,DDSDDT,DRPLDE,DRPLDT,
     2STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
     3NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
     4CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,JSTEP,KINC)
      ! Note that IMPLICIT definition is active
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME
C
      DIMENSION STRESS(NTENS),STATEV(NSTATV),
     1DDSDDE(NTENS,NTENS),DDSDDT(NTENS),DRPLDE(NTENS),
     2STRAN(NTENS),DSTRAN(NTENS),TIME(2),PREDEF(1),DPRED(1),
     3PROPS(NPROPS),COORDS(3),DROT(3,3),DFGRD0(3,3),DFGRD1(3,3),
     4JSTEP(4)
C
      ! Subroutine control
      INTEGER :: i, j, n_backstresses, it_num, converged, ind_alpha
      ! Material properties
      REAL(8) :: elastic_modulus, Q_inf, b, D_inf, a,
     1shear_modulus, bulk_modulus, poission_ratio, mu2, lame_first
      ! Used for intermediate calculations
      REAL(8) :: yield_stress, ep_eq, ep_eq_init, a_temp,
     1hard_iso_Q, hard_iso_D, hard_iso_total, a_dot_n,
     2plastic_mult, p_mult_numer, p_mult_denom, yield_function,
     3isotropic_modulus, kin_modulus,
     4stress_relative_norm, strain_trace, alpha_trace, e_k,
     5ID2_out_ID2, n_out_n, stress_hydro, sigma_vm,
     6Lam, n33_check, alpha_out_n, beta, theta_1, theta_2, theta_3,srn2
      ! Backstress arrays
      REAL(8), DIMENSION(:, :), ALLOCATABLE :: alpha_k
      REAL(8), DIMENSION(:), ALLOCATABLE :: C_k, gamma_k
      ! Tensors
      REAL(8), DIMENSION(6, 6) :: ID4, c_mat
      REAL(8), DIMENSION(6) :: strain_tens, strain_plastic,
     1yield_normal, alpha, strain_trial, stress_relative,
     2stress_dev, ID2, stress_tens, check, dstran_tens, alpha_diff,
     3alpha_upd, dpe
      ! Parameters
      INTEGER :: N_BASIC_PROPS, TERM_PER_BACK, MAX_ITERATIONS,
     1I_ALPHA
      REAL(8) :: TOL, ONE, TWO, THREE, ZERO, SQRT23
      PARAMETER(TOL=1.0D-8,
     1N_BASIC_PROPS=7, TERM_PER_BACK=2, MAX_ITERATIONS=1000,
     2ONE=1.0D0, TWO=2.0D0, THREE=3.0D0, ZERO=0.D0,
     3SQRT23=SQRT(2.0D0/3.0D0), I_ALPHA=7)
C ----------------------------------------------------------------------C
C
      ! Subroutine start
C
C ----------------------------------------------------------------------C
      ! Get the number of backstresses
      n_backstresses = (nprops - N_BASIC_PROPS) / TERM_PER_BACK
      IF (n_backstresses .EQ. 0) THEN
        PRINT *, "No backstresses defined, exiting!"
        CALL XIT  ! Exit from analysis command in Abaqus
      END IF
C
      ! Allocate the backstress related arrays
      ALLOCATE(C_k(n_backstresses))
      ALLOCATE(gamma_k(n_backstresses))
      ALLOCATE(alpha_k(n_backstresses, ntens))
C
      ! Initialize
      ddsdde(:, :) = ZERO
      ID4(:, :) = ZERO  ! 4th order symmetric identity tensor
      DO i = 1, ndi
        ID4(i, i) = ONE  
      END DO
      DO i = ndi+1, ntens
        ID4(i, i) = ONE / TWO
      END DO
      ! 2nd order symmetric identity tensor
      ID2(:) = (/ ONE, ONE, ONE, ZERO, ZERO, ZERO /)
C
      ! Read in state variables
      ! 1               = Equivalent plastic strain
      ! 2 - 7           = Plastic strain
      ! 8 - 8 + 6 * N   = Backstress components
      ep_eq = statev(1)
      ep_eq_init = statev(1)
      CALL ROTSIG(statev(2), drot, strain_plastic, 2, ndi, nshr)
      alpha(:) = ZERO
      DO i = 1, n_backstresses
        ind_alpha = I_ALPHA + 1 + (i - 1) * ntens
        CALL ROTSIG(statev(ind_alpha), drot, alpha_k(i, :), 1, ndi,nshr)
        alpha = alpha + alpha_k(i, :)
      END DO
C
      ! Read in the material properties
      elastic_modulus = props(1)
      poission_ratio = props(2)
      yield_stress = props(3)
      q_inf = props(4)
      b = props(5)
      d_inf = props(6)
      a = props(7)
      DO i = 1, n_backstresses  ! First backstress starts at index = 8
        c_k(i) = props((N_BASIC_PROPS - 1) + 2 * i)
        gamma_k(i) = props(N_BASIC_PROPS + 2 * i)
      END DO
C
      ! Calculate elastic parameters
      shear_modulus = elastic_modulus / (TWO * (ONE + poission_ratio))
      bulk_modulus = elastic_modulus /
     1(THREE * (ONE - TWO * poission_ratio))
      mu2 = TWO * shear_modulus
C
      ! Set-up strain tensor
      ! Tensors are stored: 11, 22, 33, 12, 13, 23
      strain_tens = stran + dstran
C ----------------------------------------------------------------------C
C
      ! Elastic trial step
C
C ----------------------------------------------------------------------C
      ! Tensor of elastic moduli
      DO j = 1, ntens
        DO i = 1, ntens
          ID2_out_ID2 = ID2(i) * ID2(j)
          c_mat(i, j) = ID2_out_ID2 * bulk_modulus +
     1    mu2 * (ID4(i, j) - ONE/THREE * ID2_out_ID2)
        END DO
      END DO
      ! Stress tensor
      stress_tens = stress + MATMUL(c_mat, dstran)
      !stress_tens = MATMUL(c_mat, (strain_tens - strain_plastic))
C
      stress_hydro = SUM(stress_tens(1:3)) / THREE
      strain_trace = SUM(strain_tens(1:3))
      DO i = 1, ndi
        stress_dev(i) = stress_tens(i) - stress_hydro
        stress_relative(i) = stress_dev(i) - alpha(i)
      END DO
      DO i = ndi+1, ntens
        stress_dev(i) = stress_tens(i)
        stress_relative(i) = stress_dev(i) - alpha(i)
      END DO
      stress_relative_norm = 
     1sqrt(dotprod6(stress_relative, stress_relative))
C
      ! Yield condition
      hard_iso_Q = q_inf * (ONE - EXP(-b * ep_eq))
      hard_iso_D = d_inf * (ONE - EXP(-a * ep_eq))
      hard_iso_total = yield_stress + hard_iso_Q - hard_iso_D
      yield_function = stress_relative_norm - SQRT23 * hard_iso_total
      IF (yield_function .GT. TOL) THEN
        converged = 0
      ELSE
        converged = 1
      END IF
C
      ! Calculate the normal to the yield surface
      yield_normal = stress_relative / (TOL + stress_relative_norm)
C ----------------------------------------------------------------------C
C
      ! Radial return mapping if plastic loading
C
C ----------------------------------------------------------------------C
      ! Calculate the consitency parameter (plastic multiplier)
      plastic_mult = ZERO
      it_num = 0
      DO WHILE ((converged .EQ. 0) .AND. (it_num .LT. MAX_ITERATIONS))
        it_num = it_num + 1
C
        ! Calculate the isotropic hardening parameters
        hard_iso_Q = q_inf * (ONE - EXP(-b * ep_eq))
        hard_iso_D = d_inf * (ONE - EXP(-a * ep_eq))
        hard_iso_total = yield_stress + hard_iso_Q - hard_iso_D
        isotropic_modulus = b * (q_inf - hard_iso_Q) -
     1  a * (d_inf - hard_iso_D)
        ! Calculate the kinematic hardening parameters
        kin_modulus = ZERO
        DO i = 1, n_backstresses
          e_k = EXP(-gamma_k(i) * (ep_eq - ep_eq_init))
          kin_modulus = kin_modulus + C_k(i) * e_k 
     1    - SQRT(THREE/TWO)*gamma_k(i)*e_k
     2    * dotprod6(yield_normal, alpha_k(i, :))
        END DO
        a_dot_n = ZERO
        alpha_upd(:) = ZERO
        DO i = 1, n_backstresses
          e_k = EXP(-gamma_k(i) * (ep_eq - ep_eq_init))
          alpha_upd = alpha_upd + e_k * alpha_k(i, :)
     1    + SQRT23 * C_k(i) / gamma_k(i) * (ONE - e_k) * yield_normal
        END DO
        a_dot_n = dotprod6(alpha_upd - alpha, yield_normal)  ! n : \Delta \alpha
C
        p_mult_numer = stress_relative_norm -
     1  (a_dot_n + SQRT23 * hard_iso_total + mu2 * plastic_mult)
C
        p_mult_denom = -mu2 *
     1  (ONE + (kin_modulus + isotropic_modulus) /
     2  (THREE * shear_modulus))
C
        ! Update variables
        plastic_mult = plastic_mult - p_mult_numer / p_mult_denom
        ep_eq = ep_eq_init + SQRT23 * plastic_mult
C
        IF (ABS(p_mult_numer) .LT. TOL) THEN
          converged = 1
        END IF
      END DO
C ----------------------------------------------------------------------C
C
      ! Update variables
C
C ----------------------------------------------------------------------C
      IF (it_num .EQ. 0) THEN  ! Elastic loading
        stress = stress_tens
      ELSE  ! Plastic loading
        !strain_plastic = strain_plastic + plastic_mult * yield_normal
        dpe = plastic_mult * yield_normal
        dpe(4:6) = dpe(4:6) + plastic_mult * yield_normal(4:6)
C        strain_plastic(4:6) = strain_plastic(4:6) 
C     1  + plastic_mult * yield_normal(4:6)
        strain_plastic = strain_plastic + dpe
        stress = stress_tens - MATMUL(c_mat, dpe)
        !stress = MATMUL(c_mat, (strain_tens - strain_plastic))
C
        alpha_diff = alpha
        alpha(:) = ZERO
        DO i = 1, n_backstresses  ! Update backstress components
          e_k = EXP(-gamma_k(i) * (ep_eq - ep_eq_init))
          alpha_k(i, :) = e_k * alpha_k(i, :) +
     1    SQRT23 * yield_normal * C_k(i) / gamma_k(i) * (ONE - e_k)
          alpha = alpha + alpha_k(i, :)
        END DO
        alpha_diff = alpha - alpha_diff
      END IF
C
C     Tangent modulus
      IF (it_num .EQ. 0) THEN  ! Elastic loading
        DO j = 1, ntens
          DO i = 1, ntens
            ddsdde(i, j) = c_mat(i, j)
          END DO
        END DO
        DO j = ndi+1, ntens
          ddsdde(j, j) = shear_modulus
        END DO
      ELSE  ! Plastic loading
        beta = ONE +
     1  (kin_modulus + isotropic_modulus) / (THREE * shear_modulus)
        theta_1 = ONE - mu2 * plastic_mult / stress_relative_norm
        theta_3 = ONE / (beta * stress_relative_norm)
        theta_2 = ONE / beta 
     1  + dotprod6(yield_normal, alpha_diff) * theta_3 
     2  - (ONE - theta_1)
        DO j = 1, ntens
          DO i = 1, ntens
            ID2_out_ID2 = ID2(i) * ID2(j)
            n_out_n = yield_normal(i) * yield_normal(j)
            alpha_out_n = alpha_diff(i) * yield_normal(j)
            ddsdde(i, j) = bulk_modulus * ID2_out_ID2
     1      +mu2 * theta_1*(ID4(i, j) - ONE/THREE*ID2_out_ID2)
     2      -mu2 * theta_2 * n_out_n +
     3      +mu2 * theta_3 * alpha_out_n
          END DO
        END DO
        ddsdde = ONE/TWO * (TRANSPOSE(ddsdde) + ddsdde)
      END IF
C ----------------------------------------------------------------------C
C
      ! Update the state variables
C
C ----------------------------------------------------------------------C
      statev(1) = ep_eq
      DO i = 1, ntens
        statev(i + 1) = strain_plastic(i)
      END DO
      DO i = 1, n_backstresses
        DO j = 1, ntens
          statev(I_ALPHA + j + (i-1) * ntens) = alpha_k(i, j)
        END DO
      END DO
C ----------------------------------------------------------------------C
C
      ! Reduce time increment if did not converge
C
C ----------------------------------------------------------------------C
      IF (it_num .EQ. MAX_ITERATIONS) THEN
        PRINT *, "WARNING: Return mapping in integration point ", npt,
     1  " of element ", noel, " did not converge."
        PRINT *, "Reducing time increment to 1/4 of current value."
        PNEWDT = 0.25
      END IF
      RETURN
C
      CONTAINS
C ----------------------------------------------------------------------C
C
      ! Define dot product for vectors
C
C ----------------------------------------------------------------------C
      pure function dotprod6(A, B) result(C)
      !! Returns the dot product of two symmetric length 6 vectors
      !! that are reduced from 9 components, with the last 3 symmetric
      REAL(8), intent(in) :: A(6), B(6)
      REAL(8)             :: C
      INTEGER             :: i      
      ! Calculate the dot product
      C = 0.0D0
      DO i = 1, 3
        C = C + A(i) * B(i)
      END DO
      DO i = 4, 6
        C = C + TWO * (A(i) * B(i))
      END DO
      end function
      END
C
      SUBROUTINE MPC(UE,A,JDOF,MDOF,N,JTYPE,X,U,UINIT,MAXDOF,
     * LMPC,KSTEP,KINC,TIME,NT,NF,TEMP,FIELD,LTRAN,TRAN)
C
      INCLUDE 'ABA_PARAM.INC'
C
      DIMENSION UE(MDOF),A(MDOF,MDOF,N),JDOF(MDOF,N),X(6,N),
     * U(MAXDOF,N),UINIT(MAXDOF,N),TIME(2),TEMP(NT,N),
     * FIELD(NF,NT,N),LTRAN(N),TRAN(3,3,N)
C     Internal variables
      real(8) :: disp_beam(3), rot_beam(3), w_beam, warp_fun,
     *      I33(3, 3)
      real(8) :: rmat(3, 3), t_vec(3), link(3), rotlink(3)
      integer :: i, size_field, FIELD_w, FIELD_t1, FIELD_t2, FIELD_t3
      ! Associate the field vars with their uses
      parameter(FIELD_w=1, FIELD_t1=2, FIELD_t2=3, FIELD_t3=4)
C     Constraint definition start
      ! First node is continuum, second node is beam
      disp_beam = U(1:3, 2)
      rot_beam = U(4:6, 2)
      w_beam = U(7, 2)
      warp_fun = FIELD(FIELD_w, 1, 1)
      ! Check the size for compatibility else default to z-axis
      size_field = size(FIELD, 1)
      if (size_field > 1) then
            t_vec(1) = FIELD(FIELD_t1, 1, 2)
            t_vec(2) = FIELD(FIELD_t2, 1, 2)
            t_vec(3) = FIELD(FIELD_t3, 1, 2)
      else
            t_vec = [0.d0, 0.d0, 1.d0]
      end if

      I33 = 0.d0
      do i = 1, 3
            I33(i, i) = 1.d0
      end do
      link = X(1:3, 1) - X(1:3, 2)
C     Terms independent of linear/nonlinear
      A(1:3, 1:3, 1) = I33
      A(1:3, 1:3, 2) = -I33
      do i = 1, 3
            JDOF(i, 1) = i
      end do
      do i = 1, 6
            JDOF(i, 2) = i
      end do
C     Linear constraint
      if (JTYPE == 16 .or. JTYPE == 17) then
            ! Constraint equations
            UE(1) = disp_beam(1) - link(2)*rot_beam(3)
            UE(2) = disp_beam(2) + link(1)*rot_beam(3)
            UE(3) = disp_beam(3) - link(1)*rot_beam(2) 
     *              + link(2)*rot_beam(1)
            ! Constraint linearization
            A(1:3, 4:6, 2) = skew(link)
            ! Warping component
            if (JTYPE == 17) then
                  UE(1:3) = UE(1:3) + warp_fun*w_beam*t_vec
                  A(1:3,   7, 2) = -warp_fun*t_vec
                  JDOF(7, 2) = 7
            end if
C     Nonlinear constraint
      else if (JTYPE == 26 .or. JTYPE == 27) then
            rmat = rvec2rmat(rot_beam)
            t_vec = matmul(rmat, t_vec)
            rotlink = matmul(rmat, link)
            UE(1:3) = disp_beam + rotlink - link
            A(1:3, 4:6, 2) = skew(rotlink)
            ! Warping component
            if (JTYPE == 27) then
                  UE(1:3) = UE(1:3) + warp_fun*w_beam*t_vec
                  A(1:3, 4:6, 2) = A(1:3, 4:6, 2) 
     *                   + skew(warp_fun*w_beam*t_vec)
                  A(1:3,   7, 2) = -warp_fun * t_vec
                  JDOF(7, 2) = 7
            end if
      else
            print *, 'Error in MPC Subroutine, JTYPE =', JTYPE
            print *, 'Should be: 16, 17, 26, or 27.'
            call XIT
      end if
      RETURN
C
      CONTAINS
C
C     Rotation vector to rotation matrix using Rodrigues forumla
      PURE FUNCTION RVEC2RMAT(rvec) RESULT(rrr)
      real(8), intent(in) :: rvec(3)
      real(8) :: rrr(3, 3)
      integer :: i, j
      real(8) :: r, rr(3), r_out_r(3, 3), small_tol
      parameter(small_tol=1.d-14)
      r = norm2(rvec)
      if (r < small_tol) then
            rrr = I33
      else
            rr = rvec / r
            forall (i = 1:3)
                  forall(j = 1:3) r_out_r(i, j) = rr(i) * rr(j)
            end forall
            rrr = cos(r)*I33 + (1.d0-cos(r))*r_out_r + sin(r)*skew(rr)
      end if
      END FUNCTION
C
C     Vector to skew-symmetric matrix
      PURE FUNCTION SKEW(v) RESULT(m)
      real(8), intent(in) :: v(3)
      real(8) :: m(3, 3)
      ! The following lines define the columns of the matrix
      m(1:3, 1) = [0.d0, v(3), -v(2)]
      m(1:3, 2) = [-v(3), 0.d0, v(1)]
      m(1:3, 3) = [v(2), -v(1), 0.d0]
      END FUNCTION
C
      END SUBROUTINE
C
C
      SUBROUTINE UAMP(ampName, time, ampValueOld, dt, nProps, props, nSvars, 
     1svars, lFlagsInfo, nSensor, sensorValues, sensorNames, jSensorLookUpTable,
     2ampValueNew, lFlagsDefine, AmpDerivative, AmpSecDerivative,
     3AmpIncIntegral, AmpDoubleIntegral)
C
      INCLUDE 'aba_param.inc'
C
C     Svars - additional state variables, similar to (V)UEL
      DIMENSION sensorValues(nSensor), svars(nSvars), props(nProps)
      CHARACTER*80 sensorNames(nSensor)
      CHARACTER*80 ampName
C
C     Time indices
      PARAMETER(iStepTime=1, iTotalTime=2, nTime=2)
C     Flags passed in for information
      PARAMETER(iInitialization=1, iRegularInc=2, iCuts=3, ikStep=4, nFlagsInfo=4)
C     OPTIONAL flags to be defined
      PARAMETER(iComputeDeriv=1, iComputeSecDeriv=2, iComputeInteg=3, iComputeDoubleInteg=4,
     1iStopAnalysis=5, iConcludeStep=6, nFlagsDefine=6)
      DIMENSION time(nTime), lFlagsInfo(nFlagsInfo), lFlagsDefine(nFlagsDefine)
      PARAMETER(STOP_LIMIT=4.8D-1)
C
C     Get sensor value
      RM_SENSOR  = GETSENSORVALUE('RMSENSOR', jSensorLookUpTable, sensorValues)
      DP_SENSOR  = GETSENSORVALUE('DPSENSOR', jSensorLookUpTable, sensorValues)

      IF (lFlagsInfo(iInitialization) .EQ. 1) THEN
            ampValueNew = 0.0
            lFlagsDefine(iConcludeStep) = 0
C
C     Displacement(t-1)
            svars(1) = 0.0D0
C     Reaction moment(t-1)
            svars(2) = 0.0D0
C     Maximum reaction moment
            svars(3) = 0.0D0
C
      ELSE
C
C     Update maximum positive reaction moment
            IF (RM_SENSOR .GT. svars(3)) THEN
                  svars(3) = RM_SENSOR
            END IF
C
C     Stopping criterion
            IF ((DP_SENSOR .GT. svars(1)) .and. (DP_SENSOR .GT. 0.0D0)
     1        .and. (RM_SENSOR .LT. svars(2)) .and. (RM_SENSOR .GT. 0.0D0)
     2        .and. (RM_SENSOR .LT. (svars(3) * STOP_LIMIT))) THEN
                  lFlagsDefine(iConcludeStep) = 1
            END IF
C
C     Update t-1 values
            svars(1) = DP_SENSOR
            svars(2) = RM_SENSOR
C
      END IF
C
      END SUBROUTINE