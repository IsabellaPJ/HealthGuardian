-- phpMyAdmin SQL Dump
-- version 5.1.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Oct 25, 2024 at 06:53 AM
-- Server version: 10.4.19-MariaDB
-- PHP Version: 7.3.28

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `healthguidance`
--

-- --------------------------------------------------------

--
-- Table structure for table `appointment`
--

CREATE TABLE `appointment` (
  `drid` int(100) NOT NULL,
  `drname` varchar(255) NOT NULL,
  `dremail` varchar(255) NOT NULL,
  `Specialist` varchar(255) NOT NULL,
  `drstatus` varchar(255) NOT NULL,
  `booking_status` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `appointment`
--

INSERT INTO `appointment` (`drid`, `drname`, `dremail`, `Specialist`, `drstatus`, `booking_status`) VALUES
(1, 'Stephen', 'stephen@gmail.com', 'Child Specialist', 'Available', 'Booked'),
(2, 'Zach', 'zach@gmail.com', 'Orthopedics', 'Available', 'Not Booked'),
(3, 'Joe', 'joe@gmail.com', 'Dermatologist', 'Available', 'Not Booked'),
(4, 'Steve', 'steve@gmail.com', 'ENT', 'Available', 'Booked'),
(5, 'Jobs', 'jobs@gmail.com', 'General', 'Available', 'Not Booked'),
(6, 'Dom', 'dom@gmail.com', 'Gynacologist', 'Available', 'Not Booked'),
(8, 'Bob', 'bob@gmail.com', 'Pediatrician', 'Unavailable', 'Not Booked'),
(9, 'Ell', 'ell@gmail.com', 'Opthalmology', 'Available', 'Not Booked'),
(10, 'Donny', 'donny@gmail.com', 'Radiology', 'Unavailable', 'Not Booked');

-- --------------------------------------------------------

--
-- Table structure for table `patient`
--

CREATE TABLE `patient` (
  `Id` int(11) NOT NULL,
  `name` varchar(100) NOT NULL,
  `email` varchar(100) NOT NULL,
  `phone_no` varchar(100) NOT NULL,
  `appoint_date` date NOT NULL,
  `appoint_time` time NOT NULL,
  `drid` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `patient`
--

INSERT INTO `patient` (`Id`, `name`, `email`, `phone_no`, `appoint_date`, `appoint_time`, `drid`) VALUES
(1, 'Isabella', 'isabel3112k@gmail.com', '9876543210', '2024-10-18', '10:00:00', 1),
(2, 'Isabella', '2033015mdcs@cit.edu.in', '9876543210', '2024-10-18', '10:00:00', 4);

-- --------------------------------------------------------

--
-- Table structure for table `signup`
--

CREATE TABLE `signup` (
  `username` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `mobilenumber` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `signup`
--

INSERT INTO `signup` (`username`, `password`, `email`, `mobilenumber`) VALUES
('abirami', 'Abirami@20', 'abi@gmail.com', '9003764378'),
('Tanushree', 'Tanushree@2k', 'isabel3112k@gmail.com', '9043176090'),
('radhamani', 'Radhamani@2k', 'radhamani@gmail.com', '9043176090');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `appointment`
--
ALTER TABLE `appointment`
  ADD PRIMARY KEY (`drid`);

--
-- Indexes for table `patient`
--
ALTER TABLE `patient`
  ADD PRIMARY KEY (`Id`);

--
-- Indexes for table `signup`
--
ALTER TABLE `signup`
  ADD PRIMARY KEY (`email`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `patient`
--
ALTER TABLE `patient`
  MODIFY `Id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
