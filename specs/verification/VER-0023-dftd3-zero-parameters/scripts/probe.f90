! Probe driver for dftd3-lib (links libdftd3.a, uses the library API, full double precision output).
! Commands on stdin, one per line:
!   R                          -> dump r0ab(1:94,1:94) as stored in the library (bohr), row by row
!   P <func> <tz 0/1>          -> print s6 rs6 s18 rs18 alp of setfuncpar(version=3)
!   Q <s6> <rs6> <s18> <rs18> <z1> <z2> <r_bohr>  -> same with explicit parameters
!   D <func> <tz> <z1> <z2> <r_bohr>   -> two-body dispersion energy (Hartree) of the dimer, zero damping, no ABC
! Build: gfortran -I<lib> probe.f90 <lib>/libdftd3.a -o probe
program probe
  use dftd3_api
  implicit none
  integer, parameter :: wp = kind(1.0d0)
  type(dftd3_input) :: inp
  type(dftd3_calc) :: calc
  character(len=1) :: cmd
  character(len=64) :: func
  integer :: tz, z1, z2, ios, i, j
  real(wp) :: r, e, xyz(3, 2)
  character(len=256) :: line
  call dftd3_init(calc, inp)
  do
    read (*, '(A)', iostat=ios) line
    if (ios /= 0) exit
    if (len_trim(line) == 0) cycle
    cmd = line(1:1)
    select case (cmd)
    case ('R')
      do i = 1, 94
        write (*, '(94(ES24.16,1X))') (calc%r0ab(i, j), j = 1, 94)
      end do
    case ('P')
      read (line(2:), *) func, tz
      call dftd3_set_functional(calc, trim(func), 3, tz == 1)
      write (*, '(A,1X,5(ES24.16,1X))') trim(func), calc%s6, calc%rs6, calc%s18, calc%rs18, calc%alp
    case ('Q')
      ! explicit parameters: Q s6 rs6 s18 rs18 z1 z2 r  (zero damping, alp=14)
      read (line(2:), *) calc%s6, calc%rs6, calc%s18, calc%rs18, z1, z2, r
      calc%version = 3
      calc%alp = 14.0_wp
      xyz = 0.0_wp
      xyz(3, 2) = r
      call dftd3_dispersion(calc, xyz, [z1, z2], e)
      write (*, '(ES24.16)') e
    case ('D')
      read (line(2:), *) func, tz, z1, z2, r
      call dftd3_set_functional(calc, trim(func), 3, tz == 1)
      xyz = 0.0_wp
      xyz(3, 2) = r
      call dftd3_dispersion(calc, xyz, [z1, z2], e)
      write (*, '(ES24.16)') e
    end select
  end do
end program probe
